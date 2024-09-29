#!/bin/python 
# Define the class for multi-thread process.

import concurrent.futures
import threading
import queue
import json 
import time 
from functools import partial


END_FLAG = "[END]"
class MultithreadCall:
    def __init__(self, examples, fn, output_file, num_threads=20, append=False) -> None:
        """
        Args:
            examples: a list of examples (total samples)
            fn: a function to process each example
            output_file: the output file (**jsonl**)
            num_threads: number of threads
            append: whether to append to the file
        """
        self.examples = examples
        self.output_file = output_file
        self.total = len(examples)
        self.cur = 0
        self.results_queue = queue.Queue()
        self.num_threads = num_threads
        self.file_writer = open(output_file, 'w') if not append else open(output_file, 'a')
        self.fn = fn
        self.timeout = 100 # 100s for each example
        self._init_threads()

    def _init_threads(self):
        # Create a thread for writing results to the file
        self.write_thread = threading.Thread(target=self._write_results)
        self.write_thread.start()

        # Create a thread pool executor
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads)

    def _end(self):
        # Wait for all threads to complete
        self.executor.shutdown()
        self.results_queue.put(END_FLAG)  # Add sentinel value to signal the end of results

        # Wait for the write thread to finish
        self.write_thread.join()

    def _write_results(self):
        """ Write results to a file constantly """ 
        start_time = time.time()  # Record the start time
        
        while True:
            result = self.results_queue.get()  # Retrieve result from the queue
            if result == END_FLAG:  # Terminate the thread if the sentinel value is received
                print("\nAll results have been written to the file.")
                break
            self.cur += 1
            elapsed_time = time.time() - start_time  # Calculate the elapsed time
            average_time_per_sample = elapsed_time / self.cur  # Calculate average time per sample

            print(">>>>>>>>>>>> Progress: {}/{} | Elapsed Time: {:.2f}s | Avg Time/Sample: {:.2f}s".format(
                self.cur, self.total, elapsed_time, average_time_per_sample), end='\r')

            self.file_writer.write(json.dumps(result, ensure_ascii=False) + '\n')  # Write result to the file
            self.file_writer.flush()

    def run(self):
        # Submit queries to the thread pool executor
        futures = [self.executor.submit(self.fn, example) for example in self.examples]
        # Process results as they become available
        for future in concurrent.futures.as_completed(futures):
            result = future.result(timeout=self.timeout)
            self.results_queue.put(result)  # Add result to the queue

        self._end()



if __name__ == "__main__":
    import time 

    # Define your function of processing each example
    def dummy_fn(model=None, x=None):
        """
        Args:
            Model: a dummy model
            x: sample

        Return:
            a dict of results
        """
        y = model(x)
        ret = {
            "x": x,
            "y": y
        }
        return ret

    examples = list(range(100))
    output_file = "test.jsonl"
    # Define your model
    model = lambda x: x**2
    # Define your function
    fn = partial(dummy_fn, model=model)
    caller = MultithreadCall(examples, fn, output_file, num_threads=20)
    caller.run()
    print("Done!")
