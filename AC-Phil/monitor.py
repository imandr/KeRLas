import json, csv
from webpie import WPHandler, WPApp, HTTPServer

class Monitor(object):
    
    def __init__(self, fn):
        self.FileName = fn
        self.Labels = set()
        self.Data = []          # [(t, data_dict),]
        self.SaveInterval = 1
        self.NextSave = self.SaveInterval
        
    def add(self, t, data=None, **data_args):
        if data is None:    data = data_args
        self.Data.append((t, data.copy()))
        for k in data.keys():
            self.Labels.add(k)
        self.NextSave -= 1
        if self.NextSave <= 0:
            self.save()
            self.NextSave = self.SaveInterval

    def data_as_table(self):
        labels = list(self.Labels)
        rows = []
        for t, row in self.Data:
            rows.append([t]+[row.get(l) for l in labels])
        return ['t']+labels, rows
            
    def save(self):
        labels, rows = self.data_as_table()
        with open(self.FileName, "w") as f:
            writer = csv.writer(f)
            writer.writerow(labels)
            for row in rows:
                writer.writerow(row)

class Handler(WPHandler):
    
    def data(self, request, relpath, **args):
        labels, rows = self.App.data()
        out = {
            "labels":labels,
            "data":rows
        }
        return json.dumps(out), "text/json"
        
class App(WPApp):
    
    def __init__(self, mon, **args):
        WPApp.__init__(self, Handler, **args)
        self.Monitor = mon
        
    def data(self):
        return self.Monitor.data_as_table()


def http_server(port, mon):    
    app = App(mon, static_location="static", enable_static=True)    
    return HTTPServer(port, app)