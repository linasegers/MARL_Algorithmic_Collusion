from functools import singledispatch, update_wrapper

def instance_dispatch(func):
    '''
    This function wraps the singledispatch function to make it usuable in classes
    Note from Python 3.8 onwards consider:
    https://docs.python.org/3/library/functools.html#functools.singledispatchmethod
    '''
    dispatcher = singledispatch(func)
    def wrapper(*args, **kw):
        return dispatcher.dispatch(list(kw.values())[0].__class__)(*args, **kw)
    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper