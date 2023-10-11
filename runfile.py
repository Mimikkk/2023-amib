import src.resources as resources

def main(name: str):
  resources.create(f"{name}", {"dupa": '123', "dwa": 2})
  print(resources.read(f"{name}"))

if __name__ == '__main__':
  main("test")
