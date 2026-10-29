"""Include the teammate's FedLC loss/routing/recovery tests in normal CI."""
from extensions.fedlc import test_fedlc


def load_tests(loader, tests, pattern):
    return loader.loadTestsFromModule(test_fedlc)
