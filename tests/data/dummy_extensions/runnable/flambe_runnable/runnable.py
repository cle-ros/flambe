from flambe.runnable import Runnable


class DummyRunnable(Runnable):

    def run(self, **kwargs) -> None:
        print("Dummy Runnable")
