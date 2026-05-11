"""CLI module for sparrowpy method."""
import os
from .sparrowpy_interface import sparrowpyMethod


def main() -> None:
    """Run the sparrowpy method simulation."""
    # JSON path in the uploads folder. This variable is set for the
    # container when it is started up.
    json_file_path = os.environ.get("JSON_PATH")

    print(f"Running sparrowpy method with JSON_PATH={json_file_path}")
    sparrowpy_method_object = sparrowpyMethod(json_file_path)
    sparrowpy_method_object.run_simulation()

    # Save the results to a separate file
    sparrowpy_method_object.save_results()

    print("sparrowpy container finished.")
