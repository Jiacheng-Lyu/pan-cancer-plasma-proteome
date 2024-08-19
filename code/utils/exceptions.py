class ProjectError(Exception):
    """Base class for all exceptions we'll raise"""
    pass


class ParameterError(ProjectError):
    '''Invalid patameter'''
    pass


class MethodError(ProjectError):
    '''Call wrong method'''
    pass
