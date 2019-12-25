import json, csv
from webpie import WPHandler, WPApp, HTTPServer

class Handler(WPHandler):
    
    def data(self, request, relpath, **args):
        labels, rows = self.App.data_as_table()
        out = {
            "labels":labels,
            "data":rows
        }
        return json.dumps(out), "text/json"
        
class Monitor(WPApp):
    
    def __init__(self, fn, handler=None):
        if handler is not None:
            WPApp.__init__(self, handler)
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
        labels = ['t']+list(self.Labels)
        rows = []
        for t, row in self.Data:
            rows.append([t]+[row.get(l) for l in labels])
        return labels, rows
            
    def save(self):
        labels, rows = self.data_as_table()
        with open(self.FileName, "w") as f:
            writer = csv.writer(f)
            writer.writerow(labels)
            for row in rows:
                writer.writerow(row)

def http_server(port, fn):        
    app = Monitor(fn, Handler)
    return HTTPServer(port, app)