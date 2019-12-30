import json, csv
from webpie import WPHandler, WPApp, HTTPServer

class Monitor(object):
    
    def __init__(self, fn, plot_desc = []):
        #
        # plot_desc: [ plot_dict, ... ]
        # plot_dict: { plot_label: attributes or null }
        # attributes:
        # {
        #   secondary_axis: <bool>,                 default: false
        #   line_width:     <float> or 0.0,         default: 1.0
        #   color:     <string>                default: auto
        #   marker_style:   <string> or null,       default: null
        # }
        #
        self.FileName = fn
        self.Labels = set()
        self.Data = []          # [(t, data_dict),]
        self.SaveInterval = 1
        self.NextSave = self.SaveInterval
        self.Server = None
        self.PlotDesc = plot_desc
        
    def reset(self):
        self.Labels = set()
        self.Data = []
        self.NextSave = self.SaveInterval        
        
    def start_server(self, port):
        app = App(self, static_location="static", enable_static=True)    
        self.Server = HTTPServer(port, app, logging=False)
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
        labels = sorted(list(self.Labels))
        rows = []
        n = len(self.Data)
        prescale = 1.0 if n < 10000 else 0.1
        scaler = 0.0
        for i, (t, row) in enumerate(self.Data):
            scaler += prescale
            if scaler >= 1.0 or i == n-1:
                scaler -= 1.0
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
            "labels":       labels,
            "data":         rows,
            "plot_desc":   self.App.Monitor.PlotDesc
        }
        return json.dumps(out), "text/json"
        
class App(WPApp):
    
    def __init__(self, mon, **args):
        WPApp.__init__(self, Handler, **args)
        self.Monitor = mon
        

