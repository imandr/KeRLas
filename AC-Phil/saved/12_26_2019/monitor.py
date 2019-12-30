import json, csv
from webpie import WPHandler, WPApp, HTTPServer

class Monitor(object):
    
    def __init__(self, fn, plot_attrs = {}):
        #
        # plot attributes:
        #
        # label -> 
        self.FileName = fn
        self.Labels = set()
        self.Data = []          # [(t, data_dict),]
        self.SaveInterval = 1
        self.NextSave = self.SaveInterval
        self.Server = None
        self.PlotAttributes = plot_attrs
        
    def start_server(self, port):
        app = App(self, static_location="static", enable_static=True)    
        self.Server = HTTPServer(port, app)
        self.Server.start()
        return self.Server
        
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
        labels, rows = self.App.Monitor.data_as_table()
        out = {
            "labels":labels,
            "data":rows,
            "attributes":self.App.Monitor.PlotAttributes
        }
        return json.dumps(out), "text/json"
        
class App(WPApp):
    
    def __init__(self, mon, **args):
        WPApp.__init__(self, Handler, **args)
        self.Monitor = mon
        

