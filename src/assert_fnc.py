def assertAllClose(A, B):
    shape1 = A.shape
    shape2 = B.shape
    assert shape1 == shape2, "Not equal"

    diff = A - B
    for _ in range(len(shape1)):
        diff = sum(diff)
    assert diff.item() < 1e-7, "Not equal"

def assertDictEqual(A, B):
    List_A = list(A.keys())
    List_B = list(B.keys())
    assert List_A == List_B, "Not equal"
    for a, b in zip(List_A, List_B):
        assert a == b, "Not equal"

def assertLen(A, Len):
    assert len(A) == Len, "Not equal"

def assertEmpty(A):
    assert A == {}, "Not equal"