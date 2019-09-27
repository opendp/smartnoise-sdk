import sys

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "Default text"
    print("Ran the example privacy module with text {}".format(text))
