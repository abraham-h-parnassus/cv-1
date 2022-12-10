import os.path
from threading import Thread
from tkinter import *
from tkinter import filedialog as fd
from tkinter import ttk

from PIL import Image, ImageTk
from sklearn import svm

from checker import check, find_pedestrians
from trainer import train


class Application:

    def __init__(self, root):
        self.root = root
        self.frame = ttk.Frame(root, padding=10)
        self.train_section = ttk.Labelframe(self.frame, text='Training')
        self.assess_section = ttk.Labelframe(self.frame, text='Assessment')
        self.class_section = ttk.Labelframe(self.frame, text='Classification')
        self.image_section = ttk.Labelframe(self.frame, text='Image')

        self.train_choose_dataset_btn = ttk.Button(self.train_section, text="Choose IDL file",
                                                   command=self.select_train_dataset)
        self.train_dataset_label = ttk.Label(self.train_section, text="n/a")
        self.train_choose_model_btn = ttk.Button(self.train_section, text="Choose model file",
                                                 command=self.select_train_model)
        self.train_model_label = ttk.Label(self.train_section, text="n/a")
        self.train_go_btn = ttk.Button(self.train_section, text="Go", command=self.go_train)
        self.train_model_file = None
        self.train_dataset_file = None

        self.class_choose_image_btn = ttk.Button(self.class_section, text="Choose image",
                                                 command=self.select_class_image)
        self.class_image_label = ttk.Label(self.class_section, text="n/a")
        self.class_choose_model_btn = ttk.Button(self.class_section, text="Choose model file",
                                                 command=self.select_class_model)
        self.class_model_label = ttk.Label(self.class_section, text="n/a")
        self.class_go_btn = ttk.Button(self.class_section, text="Go", command=self.go_classification)
        self.class_image_file = None
        self.class_model_file = None

        self.assess_choose_dataset_btn = ttk.Button(self.assess_section, text="Choose IDL file",
                                                    command=self.select_asssess_dataset)
        self.assess_dataset_label = ttk.Label(self.assess_section, text="n/a")
        self.assess_choose_model_btn = ttk.Button(self.assess_section, text="Choose model file",
                                                  command=self.select_assess_model)
        self.assess_model_label = ttk.Label(self.assess_section, text="n/a")
        self.assess_stats_caption_label = ttk.Label(self.assess_section, text="Precision / Recall:")
        self.assess_stats_value_label = ttk.Label(self.assess_section, text="n/a")
        self.assess_go_btn = ttk.Button(self.assess_section, text="Go", command=self.go_assess)
        self.assess_model_file = None
        self.assess_dataset_file = None

    def show(self):
        self.frame.grid()
        self.train_section.grid(column=0, row=0)
        self.class_section.grid(column=0, row=1)
        self.assess_section.grid(column=0, row=2)
        self.image_section.grid(column=1, row=0, rowspan=3)

        self.train_go_btn.grid(column=0, row=0)
        self.train_choose_dataset_btn.grid(column=1, row=0)
        self.train_dataset_label.grid(column=2, row=0)
        self.train_choose_model_btn.grid(column=3, row=0)
        self.train_model_label.grid(column=4, row=0)

        self.class_go_btn.grid(column=0, row=0)
        self.class_choose_image_btn.grid(column=1, row=0)
        self.class_image_label.grid(column=2, row=0)
        self.class_choose_model_btn.grid(column=3, row=0)
        self.class_model_label.grid(column=4, row=0)

        self.assess_go_btn.grid(column=0, row=0)
        self.assess_choose_dataset_btn.grid(column=1, row=0)
        self.assess_dataset_label.grid(column=2, row=0)
        self.assess_choose_model_btn.grid(column=3, row=0)
        self.assess_model_label.grid(column=4, row=0)
        self.assess_stats_caption_label.grid(column=0, row=1)
        self.assess_stats_value_label.grid(column=1, row=1)

        self.root.mainloop()

    def select_train_model(self):
        path = fd.asksaveasfilename()
        base_name = os.path.basename(path)
        self.train_model_label.configure(text=base_name)
        self.assess_model_label.configure(text=base_name)
        self.class_model_label.configure(text=base_name)

        self.train_model_file = path
        self.assess_model_file = path
        self.class_model_file = path

    def select_train_dataset(self):
        path = fd.askopenfile()
        base_name = os.path.basename(path.name)
        self.train_dataset_label.configure(text=base_name)
        self.train_dataset_file = path.name

    def select_class_image(self):
        path = fd.askopenfile()
        base_name = os.path.basename(path.name)
        self.class_image_label.configure(text=base_name)
        self.class_image_file = path.name

    def select_class_model(self):
        path = fd.askopenfile()
        base_name = os.path.basename(path.name)
        self.class_model_label.configure(text=base_name)
        self.class_model_file = path.name

    def select_assess_model(self):
        path = fd.askopenfile()
        base_name = os.path.basename(path.name)
        self.assess_model_label.configure(text=base_name)
        self.assess_model_file = path.name

    def select_asssess_dataset(self):
        path = fd.askopenfile()
        base_name = os.path.basename(path.name)
        self.assess_dataset_label.configure(text=base_name)
        self.assess_dataset_file = path.name

    def go_train(self):
        self.train_go_btn.configure(text="Go ⌛")
        thread = Thread(target=self._train)
        thread.start()

    def go_classification(self):
        self.class_go_btn.configure(text="Go ⌛")
        thread = Thread(target=self._classify)
        thread.start()

    def go_assess(self):
        self.assess_go_btn.configure(text="Go ⌛")
        thread = Thread(target=self._assess)
        thread.start()

    def _train(self):
        clf = svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
                      decision_function_shape='ovr', degree=3, gamma='auto', kernel="linear",
                      max_iter=-1, probability=False, random_state=None, shrinking=True,
                      tol=0.001, verbose=False)
        train(clf, self.train_dataset_file, self.train_model_file)
        self.train_go_btn.configure(text="Go ✓")

    def _assess(self):
        recall, precision, _ = check(self.assess_model_file, self.assess_dataset_file)
        self.assess_stats_value_label.configure(text=f"{precision}% / {recall}%")
        self.assess_go_btn.configure(text="Go ✓")

    def _classify(self):
        result = find_pedestrians(self.class_model_file, self.class_image_file)
        img = self._photo_image(result)
        panel = Label(self.image_section, image=img)
        panel.photo = img
        panel.grid(column=0, row=0)
        self.root.update()
        self.train_go_btn.configure(text="Go ✓")

    def _photo_image(self, image):
        return ImageTk.PhotoImage(image=Image.fromarray(image))


def show_ui():
    root = Tk()
    Application(root).show()
    root.mainloop()


if __name__ == '__main__':
    show_ui()
