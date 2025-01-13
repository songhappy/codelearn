def search(nums: [int], target: int) -> int:
    l = 0
    r = len(nums) - 1
    while l <= r:
        m = (l + r) // 2
        if nums[m] == target:
            return m
        elif nums[m] < target:
            l = m + 1
        else:
            r = m - 1
    return -1


def index_strings(names):
    names_dict = {}
    for i, name in enumerate(names):
        if name not in names_dict:
            names_dict[name] = len(names_dict) + 1
    return names_dict


if __name__ == '__main__':
    print(search(nums=[1, 2, 3, 4, 5, 6], target=5))
    print(index_strings(['Guoqiong', 'Benjamin', 'Lingyun', 'Guoqiong', "hello"]))
