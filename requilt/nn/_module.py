import warp as wp


class Module:
    def reset_parameters(self):
        pass

    def reset_kernels(self):
        pass

    def parameters(self):
        params = []
        for k, v in vars(self).items():
            if isinstance(v, wp.array):
                params.append(v)
        return params

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass


class ModuleList(Module):
    pass


class ModuleDict(Module):
    pass
