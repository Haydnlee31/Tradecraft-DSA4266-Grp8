"""Byte-serialized, bidirectional compression around persistent server Adam.

Clients differentiate the same delivered model, never run local Adam, and send
only compressed gradients. The server keeps full-precision target weights and
Adam moments. This is an empirical extension, not 1-bit Adam or an EF-SGD theorem.
"""
import copy
import struct

import numpy as np
import torch

from src.eval import shared_optimizer_performance as training
from src.federated import shared_adam as shared

HEADER = struct.Struct('<BBHI')  # version, bits, tensor count, sequence number
TENSOR = struct.Struct('<If')    # element count and FP32 scale


def payload_size(refs, bits):
    shared.require(bits in (2,4,32), 'Unsupported precision')
    return HEADER.size + sum(TENSOR.size + (p.numel()*bits+7)//8 for p in refs)


def encode(values, bits, sequence):
    """Actually pack 2/4-bit symbols; an int8 container alone saves fewer bytes.

    Each tensor has its own symmetric max-absolute scale. Round ties to even.
    The final code is reserved; zero padding is checked by the receiver.
    Tensor shapes/order belong to the pre-agreed model schema, not this packet.
    """
    shared.require(bits in (2,4,32) and type(sequence) is int and 0 <= sequence < 2**32,
                   'Invalid packet settings')
    shared.require(0 < len(values) < 2**16, 'Invalid tensor count')
    chunks = [HEADER.pack(1,bits,len(values),sequence)]
    for value in values:
        shared.require(value.device.type == 'cpu' and value.dtype == torch.float32 and
                       value.numel() > 0 and bool(torch.isfinite(value).all()), 'Invalid tensor')
        a = value.detach().contiguous().numpy().reshape(-1)
        if bits == 32:
            scale, data = 1., a.astype('<f4',copy=False).tobytes()
        else:
            qmax = 2**(bits-1)-1
            maximum = float(np.abs(a).max())
            scale = float(np.float32(max(maximum/qmax,np.finfo(np.float32).tiny))) if maximum else 1.
            q = np.clip(np.rint(a.astype(np.float64)/scale),-qmax,qmax).astype(np.int16)
            codes = (q+qmax).astype(np.uint8)
            per_byte = 8//bits
            padded = np.zeros(((len(codes)+per_byte-1)//per_byte)*per_byte,dtype=np.uint8)
            padded[:len(codes)] = codes
            blocks = padded.reshape(-1,per_byte)
            packed = np.zeros(len(blocks),dtype=np.uint8)
            for j in range(per_byte):
                packed |= blocks[:,j] << (j*bits)
            data = packed.tobytes()
        chunks.extend((TENSOR.pack(len(a),scale),data))
    return b''.join(chunks)


def decode(packet, refs, bits, sequence):
    """Only decoded payloads reach aggregation or receiver model updates."""
    shared.require(type(packet) is bytes and len(packet) == payload_size(refs,bits), 'Packet length differs')
    shared.require(HEADER.unpack_from(packet) == (1,bits,len(refs),sequence), 'Packet header/sequence differs')
    offset = HEADER.size
    values = []
    for ref in refs:
        n,scale = TENSOR.unpack_from(packet,offset)
        offset += TENSOR.size
        shared.require(n == ref.numel() and np.isfinite(scale) and scale > 0, 'Tensor schema/scale differs')
        length = (n*bits+7)//8
        raw = packet[offset:offset+length]
        offset += length
        if bits == 32:
            shared.require(scale == 1., 'FP32 scale differs')
            a = np.frombuffer(raw,dtype='<f4').copy()
        else:
            packed = np.frombuffer(raw,dtype=np.uint8)
            codes = np.empty(len(packed)*(8//bits),dtype=np.uint8)
            for j in range(8//bits):
                codes[j::8//bits] = (packed >> (j*bits)) & (2**bits-1)
            shared.require(bool((codes[n:] == 0).all()) and bool((codes[:n] < 2**bits-1).all()),
                           'Invalid padding/reserved symbol')
            a = (codes[:n].astype(np.int16)-(2**(bits-1)-1)).astype(np.float32)*np.float32(scale)
        shared.require(bool(np.isfinite(a).all()), 'Nonfinite decoded values')
        values.append(torch.from_numpy(a.reshape(tuple(ref.shape))))
    return tuple(values)


def make_state(base, bits, feedback, lr):
    state = training.make_state(base,'shared_adam',lr)
    state['visible'] = copy.deepcopy(state['model'])
    ps = shared.parameters(state['model'])
    initial = encode(ps,32,0)
    with torch.no_grad():
        for p,q in zip(state['visible'].parameters(),decode(initial,ps,32,0)):
            p.copy_(q)
    state['uplink_errors'] = [tuple(torch.zeros_like(p) for p in ps) for _ in range(base['clients'])]
    state['bits'],state['feedback'] = bits,feedback
    state['wire'] = {'initialization_bytes':base['clients']*len(initial),'uplink_bytes':0,'downlink_bytes':0}
    state['work']['estimated_payload_bytes'] = state['wire']['initialization_bytes']
    return state


def advance(state, data, base, lr, criterion):
    bits,feedback = state['bits'],state['feedback']
    model,visible = state['model'],state['visible']
    model.train(); visible.train()
    ps = shared.parameters(model)
    unit = state['unit']
    round_id,step = divmod(unit,base['mime']['local_steps'])
    messages = []
    for i in range(base['clients']):
        rows = training.sampled_rows(data,base,round_id,i,step)
        training.record_rows(state,rows)
        gradient = shared.client_gradient(visible,data.batch(rows),criterion,base['seed'])
        corrected = tuple(g+e for g,e in zip(gradient,state['uplink_errors'][i])) if feedback and bits != 32 else gradient
        packet = encode(corrected,bits,unit+1)
        decoded = decode(packet,ps,bits,unit+1)
        state['uplink_errors'][i] = (tuple(g-q for g,q in zip(corrected,decoded)) if feedback and bits != 32
                                     else tuple(torch.zeros_like(g) for g in gradient))
        messages.append(decoded)
        state['wire']['uplink_bytes'] += len(packet)
    # Adam retains its full-precision target and moments. Gradients come from
    # the delivered approximation, so this is not mathematically exact Adam.
    previous = tuple(p.detach().clone() for p in ps)
    shared.apply_gradient(model,state['optimizer'],shared.aggregate_gradients(messages,[len(p) for p in data.parts]))
    if bits == 32:
        # Absolute FP32 weights avoid subtract/add roundoff in the identity bridge.
        payload = tuple(p.detach() for p in ps)
    elif feedback:
        # target - last delivered model = new Adam update + undelivered error.
        payload = tuple(p.detach()-v.detach() for p,v in zip(ps,visible.parameters()))
    else:
        payload = tuple(p.detach()-old for p,old in zip(ps,previous))
    packet = encode(payload,bits,unit+1)
    decoded = decode(packet,ps,bits,unit+1)
    with torch.no_grad():
        for v,q in zip(visible.parameters(),decoded):
            v.copy_(q) if bits == 32 else v.add_(q)
    shared.require(all(bool(torch.isfinite(p).all()) for p in visible.parameters()), 'Nonfinite delivered model')
    state['wire']['downlink_bytes'] += base['clients']*len(packet)
    w = state['work']
    w['backward_examples'] += base['clients']*base['batch_size']
    w['backward_batches'] += base['clients']
    w['optimizer_steps'] += 1
    w['synchronization_rounds'] += 1
    w['logical_barriers'] += 1
    w['estimated_payload_bytes'] = sum(state['wire'].values())
    state['unit'] += 1


def snapshot(state):
    return {**training.snapshot(state), 'visible':copy.deepcopy(state['visible'].state_dict()),
            **{k:copy.deepcopy(state[k]) for k in ('uplink_errors','wire','bits','feedback')}}


def restore(saved, base, lr):
    state = make_state(base,saved['bits'],saved['feedback'],lr)
    restored = training.restore(saved,base,'shared_adam',lr)
    state.update(restored)
    state['visible'].load_state_dict(saved['visible'])
    for k in ('uplink_errors','wire','bits','feedback'):
        state[k] = copy.deepcopy(saved[k])
    return state


def expected_wire(model, base, bits, units):
    ps = shared.parameters(model)
    c = base['clients']
    return {'initialization_bytes':c*payload_size(ps,32),
            'uplink_bytes':c*units*payload_size(ps,bits),
            'downlink_bytes':c*units*payload_size(ps,bits)}
