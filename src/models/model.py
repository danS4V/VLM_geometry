from functools import reduce
from collections.abc import Callable

def get_attr_or_item(obj, attr):
        '''
        Access nested attributes or indexed items in a PyTorch module flexibly.
        Allows for unified access to both object attributes and ModuleList indices
        using a single interface.

        Args:
            obj: The object to access (typically a PyTorch module or submodule)
            attr (str): The attribute name or index to access. If the string represents 
                      a number (e.g., '0', '1', '28'), it will be used as an index.
                      Otherwise, it will be used as an attribute name.

        Returns:
            The accessed attribute or indexed item
        '''
        try:
            index = int(attr)  # Try to convert the attribute to an integer
            return obj[index]  # If successful, use it as an index
        except ValueError:
            # If conversion fails, treat it as a regular attribute name
            return getattr(obj, attr)

class Model:
    def __init__(self):
      pass    

    def register_hooks(self,hook_generator : Callable[[str,...],Callable],hook_layers : list, hook_generator_kwargs={}):
        '''Register forward hooks for specified layers.
        self.probe_layers has to be a list of either str or int.
        
        hook_generator
          function (layer_name, kwargs) -> funtion(model, input,output). 
        hook_layers
          A dictionary, keys are labels for the layers and values are the 'path' to the layer in the model.
          Keys are passed to hook_generator as layer_name argument.
        hook_generator_kwargs
          Optional arguments for hook_generator, usually at least a dictionary where the outputs will be saved.
        
        Returns:

        handles : dict
          hook handles, useful e.g. to remove the hooks.
        '''
        handles = {}
        for layer_name, layer_path in hook_layers.items():
            if isinstance(layer_path, str): #doesn't work with sth like '.boh[4]'
              layer_path = layer_path.split('.')
            target = reduce(get_attr_or_item, layer_path, self.model)
            handles[layer_name] = target.register_forward_hook(hook_generator(layer_name, **hook_generator_kwargs))
        
        return handles