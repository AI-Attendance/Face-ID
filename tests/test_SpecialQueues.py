import pytest
from SpecialQueues import SignalingQueue
import multiprocessing as mp


def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(argnames, [[funcargs[name] for name in argnames]
                                    for funcargs in funcarglist])


class TestSignalingQueue:

    params = {
        "test_simpleReadWriteScenario":
        [dict(data=[1, 2.2, "Queue"]),
         dict(data=None)],
        "test_simpleSignalingReadWriteScenario":
        [dict(data1="Signaling", data2="Queue", data3="Dummy")],
        "test_NReadersMWriters": [
            dict(readers=1, writers=1, data_count=3),
            dict(readers=1, writers=3, data_count=3),
            dict(readers=3, writers=1, data_count=9),
            dict(readers=3, writers=3, data_count=15)
        ]
    }

    def test_simpleReadWriteScenario(self, data):
        sq = SignalingQueue()
        assert sq.put(data) is True, "Putting error"
        assert sq.get() == data, "Received different data"

    def test_simpleSignalingReadWriteScenario(self, data1, data2, data3):
        sq = SignalingQueue()
        sq.put(data1)
        sq.put(data2)
        assert sq.get() == data1, "Received different data"
        sq.signal()
        assert sq.signaled() is True, "Signaling error"
        assert sq.qsize() == 1, "An element is still there"
        assert sq.empty() is False, "An element is still there"
        assert sq.any() is True, "An element is still there"
        assert sq.get(
        ) == data2, "Last element in the queue should be retrieved"
        assert sq.get() is None, "Queue is already signaled and empty"
        assert sq.put(data3) is False, "Shouldn't put this"

    def _reader(queue_writer_reader, queue_reader_main):
        while True:
            num = queue_writer_reader.get()
            if num is None:
                break
            queue_reader_main.put(num)

    def _writer(queue_writer_reader, start_count, end_count):
        for num in range(start_count, end_count):
            queue_writer_reader.put(num)

    def test_NReadersMWriters(self, readers, writers, data_count):
        queue_writer_reader = SignalingQueue(MultipleReaders=readers > 1,
                                             MultipleWriters=writers > 1)
        p_writers = [
            mp.Process(target=TestSignalingQueue._writer,
                       args=(queue_writer_reader, index * data_count,
                             (index + 1) * data_count))
            for index in range(writers)
        ]
        for writer in p_writers:
            writer.start()
        queue_reader_main = SignalingQueue(MultipleWriters=readers > 1)
        p_readers = [
            mp.Process(target=TestSignalingQueue._reader,
                       args=(queue_writer_reader, queue_reader_main))
            for _ in range(readers)
        ]
        for reader in p_readers:
            reader.start()
        for writer in p_writers:
            writer.join()
        queue_writer_reader.signal()
        for reader in p_readers:
            reader.join()
        queue_reader_main.signal()
        values = {}
        while True:
            num = queue_reader_main.get()
            if num is None:
                break
            values[num] = values.setdefault(num, 0) + 1
        for num in range(data_count * writers):
            assert num in values and values[
                num] == 1, f"{num} should be inserted once"
