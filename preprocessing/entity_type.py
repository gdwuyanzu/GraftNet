
subgraph_dir = "freebase_2hops/stagg.neighborhoods/"

entity_type_map = {}
def get_entity_type(file, entity):
    with open(file) as f:
        for line in f:
            try:
                e1, rel, e2 = line.strip().split(None, 2)
            except ValueError:
                continue
            