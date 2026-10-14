"""Four-bit block scales with the frozen bidirectional error-feedback semantics.

Only the scale granularity changes. Tensor mode delegates to the old serializer
and is replayed as an exact compatibility control. Historical code is immutable.
"""
import struct

import numpy as np
import torch

from src.eval import shared_optimizer_performance as training
from src.federated import compressed_shared_adam as original
from src.federated import shared_adam as shared

HEADER = original.HEADER
COUNT = struct.Struct('<I')
BLOCK_SIZE = 128


def payload_size(refs, bits, block_size):
    if block_size is None:
        return original.payload_size(refs,bits)
    shared.require(bits == 4 and block_size == BLOCK_SIZE, 'Only four-bit block128 is frozen')
    return HEADER.size + sum(4 + 4*((p.numel()+127)//128) + (p.numel()+1)//2 for p in refs)


def encode(values, bits, sequence, block_size):
    """Version 2 fixes block length at 128; blocks never cross tensor boundaries.

    Each tensor sends its element count, all FP32 block scales, then packed
    four-bit symbols. Two symbols per byte, low nibble first, with zero padding.
    Symmetric levels -7..7 use codes 0..14; code 15 is reserved.
    """
    if block_size is None:
        return original.encode(values,bits,sequence)
    shared.require(bits == 4 and block_size == BLOCK_SIZE, 'Only four-bit block128 is frozen')
    shared.require(type(sequence) is int and 0 <= sequence < 2**32 and
                   0 < len(values) < 2**16, 'Invalid packet settings')
    chunks = [HEADER.pack(2,4,len(values),sequence)]
    for value in values:
        shared.require(value.device.type == 'cpu' and value.dtype == torch.float32 and
                       value.numel() > 0 and bool(torch.isfinite(value).all()), 'Invalid tensor')
        a = value.detach().contiguous().numpy().reshape(-1)
        blocks = (len(a)+127)//128
        padded = np.zeros(blocks*128,dtype=np.float32)
        padded[:len(a)] = a
        maxima = np.abs(padded.reshape(blocks,128)).max(axis=1).astype(np.float64)
        scales = np.maximum(maxima/7,np.finfo(np.float32).tiny).astype(np.float32)
        scales[maxima == 0] = 1.
        q = np.clip(np.rint(a.astype(np.float64)/np.repeat(scales,128)[:len(a)]),-7,7).astype(np.int16)
        symbols = np.zeros(((len(a)+1)//2)*2,dtype=np.uint8)
        symbols[:len(a)] = (q+7).astype(np.uint8)
        packed = symbols[::2] | (symbols[1::2] << 4)
        chunks.extend((COUNT.pack(len(a)),scales.astype('<f4',copy=False).tobytes(),packed.tobytes()))
    return b''.join(chunks)


def decode(packet, refs, bits, sequence, block_size):
    if block_size is None:
        return original.decode(packet,refs,bits,sequence)
    shared.require(type(packet) is bytes and len(packet) == payload_size(refs,bits,block_size),
                   'Packet length differs')
    shared.require(HEADER.unpack_from(packet) == (2,4,len(refs),sequence), 'Packet header/sequence differs')
    offset = HEADER.size
    values = []
    for ref in refs:
        n, = COUNT.unpack_from(packet,offset)
        offset += COUNT.size
        shared.require(n == ref.numel() and n > 0, 'Tensor schema differs')
        blocks = (n+127)//128
        scales = np.frombuffer(packet,dtype='<f4',count=blocks,offset=offset)
        offset += blocks*4
        shared.require(bool(np.isfinite(scales).all() and (scales > 0).all()), 'Invalid block scale')
        packed = np.frombuffer(packet,dtype=np.uint8,count=(n+1)//2,offset=offset)
        offset += (n+1)//2
        codes = np.empty(len(packed)*2,dtype=np.uint8)
        codes[::2],codes[1::2] = packed & 15,packed >> 4
        shared.require(bool((codes[n:] == 0).all() and (codes[:n] < 15).all()),
                       'Invalid padding/reserved symbol')
        a = (codes[:n].astype(np.int16)-7).astype(np.float32)*np.repeat(scales,128)[:n]
        shared.require(bool(np.isfinite(a).all()), 'Nonfinite decoded values')
        values.append(torch.from_numpy(a.reshape(tuple(ref.shape))))
    return tuple(values)


def make_state(base, bits, feedback, lr, block_size):
    shared.require(bits == 4 and feedback is True and block_size in (None,128), 'Unfrozen settings')
    # Initialization remains exactly the old full-precision broadcast.
    state = original.make_state(base,bits,feedback,lr)
    state['block_size'] = block_size
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
        packet = encode(corrected,bits,unit+1,state['block_size'])
        decoded = decode(packet,ps,bits,unit+1,state['block_size'])
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
    packet = encode(payload,bits,unit+1,state['block_size'])
    decoded = decode(packet,ps,bits,unit+1,state['block_size'])
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
    return {**original.snapshot(state), 'block_size':state['block_size']}


def restore(saved, base, lr):
    shared.require(saved['bits'] == 4 and saved['feedback'] is True and
                   saved['block_size'] in (None,128), 'Unfrozen saved settings')
    state = original.restore(saved,base,lr)
    state['block_size'] = saved['block_size']
    return state


def expected_wire(model, base, bits, units, block_size):
    ps = shared.parameters(model)
    c = base['clients']
    return {'initialization_bytes':c*original.payload_size(ps,32),
            'uplink_bytes':c*units*payload_size(ps,bits,block_size),
            'downlink_bytes':c*units*payload_size(ps,bits,block_size)}
