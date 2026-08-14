from .split_candidate import SplitCandidate

class MaxHeap:
    """
    USAGE: heap = MaxHeap(key=lambda candidate: candidate.gain)
    """

    def __init__(self, key):
        self._data = []
        self._key = key

    def push(self, item) -> None:
        self._data.append(item)
        self._sift_up(len(self._data) - 1)

    def pop(self):
        if not self._data:
            raise IndexError("Heap is empty")

        if (len(self._data) == 1):
            return self._data.pop()

        root = self._data[0]
        self._data[0] = self._data.pop()
        self._sift_down(0)

        return root

    def peek(self):
        if not self._data:
            raise IndexError("Heap is empty")
        return self._data[0]

    def is_empty(self):
        return len(self._data) == 0

    def __len__(self):
        return len(self._data)

    def _priority(self, candidate: SplitCandidate):
        return candidate.gain

    def _higher_priority(self, a, b) -> bool:
        return self._key(a) > self._key(b)

    def _sift_up(self, idx: int) -> None:
        while idx > 0:
            parent = (idx - 1) // 2

            if self._data[parent].gain >= self._data[idx].gain:
                break

            self._data[parent], self._data[idx] = self._data[idx], self._data[parent]

            idx = parent

    def _sift_down(self, idx: int) -> None:
        n = len(self._data)

        while True:
            left = 2 * idx + 1
            right = 2 * idx + 2
            largest = idx

            if left < n and self._data[left].gain > self._data[largest].gain:
                largest = left
            if right < n and self._data[right].gain > self._data[largest].gain:
                largest = right

            if largest == idx:
                break

            self._data[idx], self._data[largest] = self._data[largest], self._data[idx]

            idx = largest