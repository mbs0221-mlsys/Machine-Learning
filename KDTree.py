def variance(values, axis):
    sum = 0
    for value in values:
        sum += value[axis]
    mean = sum / len(values)
    var = 0
    for value in values:
        var += (value[axis]-mean)*(value[axis]-mean)
    var /= len(values)
    return var


def build(data_set):
    """
    build K-D Tree

    :param data_set:
    :return:
    """

    if len(data_set) is None:
        return {}

    data_set = list(data_set)

    # choose best split axis
    axis = 0
    max_var = 0
    for idx in range(len(data_set[0])):
        var = variance(data_set, idx)
        if var > max_var:
            axis = idx

    # sort points according to split axis
    data_set.sort(key=lambda x: x[axis])
    median = len(data_set)/2

    # choose median on axis
    loc = data_set[median:median+1]

    # build tree recursively
    left = build(data_set[:median])
    right = build(data_set[median+1:])

    return {loc, axis, left, right}


