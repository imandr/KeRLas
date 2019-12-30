import json, csv
from webpie import WPHandler, WPApp, HTTPServer

class Monitor(object):
    
    def __init__(self, fn, plot_attrs = {}):
        #
        # plot attributes:
        #
        # label -> <line style>
        # line style:  <line thickness>[<marker style>][#<color>]]
        # line thickness: <space> _ - ~ =
        # color: (<name>|hex)
        # marker style: <space> + . *
        self.FileName = fn
        self.Labels = set()
        self.Data = []          # [(t, data_dict),]
        self.SaveInterval = 1
        self.NextSave = self.SaveInterval
        self.Server = None
        self.PlotAttributes = self.parse_plot_attributes(plot_attrs)
        
    def reset(self):
        self.Labels = set()
        self.Data = []
        self.NextSave = self.SaveInterval        
        
    def parse_plot_attributes(self, plot_attrs):
        parsed = {}
        for label, spec in plot_attrs.items():
            line_thickness = color = marker_style = None
            marker_visizble = False
            words = spec.split("#", 1)
            line_spec = words[0][:1] or ' '
            marker_spec = words[0][1:] or ' '
            line_thickness = {" ":0.0, "_": 0.1, "-":0.5, "~":1.0, "=":2.0}[line_spec[0]]
            line_visible = line_thickness > 0
            
            if marker_spec != ' ':
                marker_visizble = True
                marker_style = {".": "circle", "*":"star", "+":"square"}[marker_spec]
            
            if len(words) > 1:
                color = words[1]
                if color[0].upper() in "0123456789ABCDEF":
                    color = "#" + color
            
            parsed[label] = {
                "color":            color,
                "line_visible":     line_visible,
                "line_thickness":   line_thickness,
                "marker_visible":   marker_visible,
                "marker_style":     marker_style
            }
        return parsed
            
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
            "labels":labels,
            "data":rows,
            "attributes":self.App.Monitor.PlotAttributes
        }
        return json.dumps(out), "text/json"
        
class App(WPApp):
    
    def __init__(self, mon, **args):
        WPApp.__init__(self, Handler, **args)
        self.Monitor = mon
        

