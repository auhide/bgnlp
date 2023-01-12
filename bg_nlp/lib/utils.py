
def make_actions_pipeline(action_list):

    def registry(func):
        action_list.append(func)
        return func

    return registry
