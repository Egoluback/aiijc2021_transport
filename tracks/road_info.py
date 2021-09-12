import osmium
import copy


class Road_info(osmium.SimpleHandler):
    def __init__(self):
        super(Road_info, self).__init__()
        self.ways = {'maxspeed': [], 'lanes': [], 'id': [], 'road_type': []}
        self.nodes = {'loc': [], 'traffic_signals': [], 'id': []}

    def node(self, n):
        tags = n.tags
        if 'highway' in tags:
            traffic_signals = copy.deepcopy(
                tags.get('highway') == 'traffic_signals')
            loc = copy.copy((n.location.lat, n.location.lon))
            id = n.id
            self.nodes['traffic_signals'].append(traffic_signals)
            self.nodes['loc'].append(loc)
            self.nodes['id'].append(id)

    def check(self, name, tags):
        var = None
        if name in tags:
            var = tags[name]
        return var

    def way(self, w):
        tags = w.tags

        if 'highway' in tags:
            road_type = tags.get('highway')
            maxspeed = self.check('maxspeed', tags)
            lanes = self.check('lanes', tags)
            self.ways['maxspeed'].append(maxspeed)
            self.ways['lanes'].append(lanes)
            self.ways['road_type'].append(road_type)
            ids = []
            for node in w.nodes:
                ids.append(node.ref)
            self.ways['id'].append(ids)

if __name__ == 'main':
    import pandas as pd

    handler = Road_info()
    path = './data/moscow.osm.pbf'
    handler.apply(path)

    ways = pd.DataFrame(handler.ways)
    ways = ways.explode(column='id', ignore_index=True).set_index('id')
    nodes = pd.DataFrame(handler.nodes).set_index('id')

    dataset = pd.merge(ways, nodes, right_index= True, left_index=True)
    dataset = dataset[~dataset.index.duplicated()]
    dataset.to_csv('./data/road_info.csv')