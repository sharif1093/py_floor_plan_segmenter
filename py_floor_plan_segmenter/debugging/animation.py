from collections import OrderedDict
from py_floor_plan_segmenter.debugging.visualization import is_notebook

try:
    import matplotlib
    from matplotlib import pyplot as plt
    import matplotlib.animation as animation
    if not is_notebook():
        matplotlib.use('Agg')
except ModuleNotFoundError:
    # Error handling
    pass


class FrameStacker:
    def __init__(self, frame_keys):
        self.frame_keys = frame_keys
        self.frames = OrderedDict({})
        self.meta = []
        self.counter = 0

    def add(self, frame, meta={}):
        self.counter += 1
        self.meta += [meta]
        for f in self.frame_keys:
            if not f in frame:
                raise KeyError("{f} does not exist in frame".format(f=f))

            if f in self.frames:
                self.frames[f] += [frame[f]]
            else:
                self.frames[f] = [frame[f]]


class FrameStackerToFile:
    def __init__(self, fs: FrameStacker, rows: int, cols: int):
        self.fs = fs
        self.forward = True

        if (rows * cols) < len(self.fs.frame_keys.keys()):
            raise ValueError("The grid with {rows} and {cols} cannot fit all frames!".format(
                rows=rows, cols=cols))

        self.axes = OrderedDict({})
        self.fig = plt.figure()
        for index, key in enumerate(self.fs.frame_keys.keys()):
            self.axes[key] = self.fig.add_subplot(rows, cols, index+1)

        self.image_handler = {}
        for key in self.axes:
            self.axes[key].set_title(self.fs.frame_keys[key]["title"])
            self.image_handler[key] = self.axes[key].imshow(fs.frames[key][0])

    def animate(self, n):
        # Reverse order when hit the end
        if (n == self.fs.counter):
            self.forward = False
        if not self.forward:
            # Go in reverse direction
            n = (2 * self.fs.counter - 1) - n

        self.fig.suptitle(self.fs.meta[n]["title"], fontsize=14)

        image_handlers_list = []
        for key in self.image_handler:
            self.image_handler[key].set_data(self.fs.frames[key][n])
            image_handlers_list += [self.image_handler[key]]

        return image_handlers_list

    def process(self, filename):
        anim = animation.FuncAnimation(
            self.fig, self.animate, interval=30, blit=True, frames=self.fs.counter * 2 - 1)

        anim.save(filename, dpi=300, writer=animation.FFMpegWriter(fps=25))
