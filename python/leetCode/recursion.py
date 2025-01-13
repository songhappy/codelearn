def get_sum(arr):
    def recursion(arr, i):
        if i < 1: return arr[i]
        the_sum = arr[i] + recursion(arr, i-1)
        return the_sum

    if len(arr) == 0: return 0
    n = len(arr) - 1
    return recursion(arr, n)


def combination(arr, target):
    # return list of list of indexes
    def recursion(arr, i, target):
        if i < 1:
            return
        diff = target - arr[i]
        return recursion(i-1, diff)

    n = len(arr) - 1
    return recursion(arr, n, target)


if __name__ == '__main__':
    res = get_sum([1,2,3,4,5])
    print(res)