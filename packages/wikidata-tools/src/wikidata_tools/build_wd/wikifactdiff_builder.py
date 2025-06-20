# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

from __future__ import annotations
import re
import sys
import threading
import time
from typing import Iterable
from textual.app import App, ComposeResult
from textual.driver import Driver
from textual.widgets import Log
from textual.containers import Horizontal
from textual.timer import Timer
from textual.widgets import RadioButton, RadioSet
from ..build_wd.config import STORAGE_FOLDER
import os.path as osp
import subprocess as sp

progress_folder = osp.join(STORAGE_FOLDER, "progress")
root_folder = osp.join(osp.dirname(__file__), "../")

WORKING_ANIMATIONS = "⣾⣽⣻⢿⡿⣟⣯⣷"
lock_progress_file = threading.Lock()


class Task:
    def __init__(
        self, name: str, command: str, dependencies: Iterable[Task] = [], buf_len=10000
    ) -> None:
        self.name = name
        self.command = command
        self.dependencies = dependencies
        self.buffer = bytearray()
        self.buf_len = buf_len
        self.finished = False
        self.waiting = True
        self.retry = False

    def format_filename(self):
        return self.name.replace(" ", "_")

    def run(self):
        # This should be run in a thread
        try:
            with open(
                osp.join(progress_folder, self.format_filename() + ".success")
            ) as f:
                list_commands = [x.strip() for x in f.readlines()]
                if self.command in list_commands:
                    self.buffer.extend(b"Task already done.")
                    self.waiting = False
                    self.finished = True
                    return
        except FileNotFoundError:
            pass
        while any(not x.finished for x in self.dependencies):
            time.sleep(5)
        self.waiting = False
        while True:
            cmd = "%s %s" % (sys.executable, self.command)
            cmd = re.sub(r" +", " ", cmd).split(" ")
            log_file = open(
                osp.join(STORAGE_FOLDER, "errors", "%s.log" % self.format_filename()),
                "ab",
            )
            p = sp.Popen(
                cmd,
                cwd=root_folder,
                shell=False,
                bufsize=10,
                stdout=sp.PIPE,
                stderr=log_file,
                stdin=None,
            )
            while p.poll() is None:
                chunk = p.stdout.read(10)
                self.buffer.extend(chunk)
                self.buffer = self.buffer[-self.buf_len :]
            return_code = p.wait()

            if return_code == 0:
                break
            self.retry = True
        with lock_progress_file:
            success_path = osp.join(progress_folder, self.format_filename() + '.success')
            try:
                list_commands = [x.strip() for x in open(success_path)]
            except FileNotFoundError:
                list_commands = []
            if self.command not in list_commands:
                list_commands.append(self.command)
            with open(success_path, 'w') as f:
                f.write('\n'.join(list_commands))
        self.finished = True
        log_file.close()


class TaskLauncher:
    def __init__(self, tasks: list[tuple]) -> None:
        self.tasks: list[Task] = []
        dependencies = {}
        self.name2task = {}
        self.threads = []
        for name, command, depends in tasks:
            obj = Task(name, command)
            self.tasks.append(obj)
            dependencies[name] = depends
            self.name2task[name] = obj

        for task in self.tasks:
            task.dependencies = [self.name2task[x] for x in dependencies[task.name]]

    def run(self):

        for task in self.tasks:
            th = threading.Thread(target=Task.run, args=(task,))
            th.start()
            self.threads.append(th)


tasks = [
    (
        "Download Old Wikidata",
        "build_wd/wikidata_scripts/download_dump.py --version old",
        [],
    ),
    (
        "Download New Wikidata",
        "build_wd/wikidata_scripts/download_dump.py --version new",
        [],
    ),
    (
        "Old Wikipedia pageviews",
        "build_wd/wikidata_scripts/create_database_wikipedia_consultation.py --version old",
        [],
    ),
    (
        "New Wikipedia pageviews",
        "build_wd/wikidata_scripts/create_database_wikipedia_consultation.py --version new",
        [],
    ),
    (
        "Push Old Wikidata",
        "build_wd/wikidata_scripts/process_json_dump.py --version old",
        ["Download Old Wikidata"],
    ),
    (
        "Push New Wikidata",
        "build_wd/wikidata_scripts/process_json_dump.py --version new",
        ["Download New Wikidata"],
    ),
    (
        "Preprocess Old Wikidata",
        "build_wd/wikidata_scripts/preprocess_dump.py --version old",
        ["Push Old Wikidata"],
    ),
    (
        "Preprocess New Wikidata",
        "build_wd/wikidata_scripts/preprocess_dump.py --version new",
        ["Push New Wikidata"],
    ),
    (
        "Collect aliases (Old Wikidata)",
        "build_wd/wikidata_scripts/collect_aliases.py --version old",
        ["Preprocess Old Wikidata"],
    ),
    (
        "Collect aliases (New Wikidata)",
        "build_wd/wikidata_scripts/collect_aliases.py --version new",
        ["Preprocess New Wikidata"],
    ),
    (
        "Setup K-Nearest-Triples (KNT)",
        "build_wd/wikidata_scripts/setup_knn.py",
        ["Preprocess Old Wikidata", "Preprocess New Wikidata"],
    ),
    (
        "Compute Wikidata difference",
        "build_wd/wikidata_scripts/compute_diff.py",
        ["Preprocess Old Wikidata", "Preprocess New Wikidata"],
    ),
    (
        "Compute popularity (Old Wiki)",
        "build_wd/wikidata_scripts/compute_importance.py --version old",
        ["Old Wikipedia pageviews", "Compute Wikidata difference"],
    ),
    (
        "Compute popularity (New Wiki)",
        "build_wd/wikidata_scripts/compute_importance.py --version new",
        ["New Wikipedia pageviews", "Compute Wikidata difference"],
    ),
    (
        "Create WikiFactDiff (Triples)",
        "build_wd/wikidata_scripts/create_wikifactdiff_triples.py",
        ["Compute popularity (New Wiki)", "Compute Wikidata difference"],
    ),
    (
        "Verbalize WikiFactDiff + KNT",
        "build_wd/verbalize_wikifactdiff/verbalize_wikifactdiff.py --ann_method sparse --force_reset",
        [
            "Create WikiFactDiff (Triples)",
            "Setup K-Nearest-Triples (KNT)",
            "Collect aliases (Old Wikidata)",
        ],
    ),
    (
        "Clean WikiFactDiff",
        "build_wd/wikidata_scripts/clean_wikifactdiff.py",
        ["Verbalize WikiFactDiff + KNT"],
    ),
]


class DockLayoutExample(App):
    CSS_PATH = osp.join(root_folder, "build_wd/style.tcss")
    refresh_timer: Timer
    exit_timer: Timer
    animation_timer: Timer
    # BINDINGS = [("q", "quit", "Quit")]

    def __init__(
        self,
        task_launcher: TaskLauncher,
        tasks,
        driver_class: type[Driver] | None = None,
        css_path: str | None = None,
        watch_css: bool = False,
    ):
        super().__init__(driver_class, css_path, watch_css)
        self.tasks = tasks
        self.task_launcher = task_launcher

    def compose(self) -> ComposeResult:
        with Horizontal():
            with RadioSet(id="sidebar"):
                btn = RadioButton(self.tasks[0][0], value=True)
                btn.id_ = self.tasks[0][0]
                yield btn
                for name, _, _ in self.tasks[1:]:
                    btn = RadioButton(name)
                    btn.id_ = name
                    yield btn
            yield Log(id="content")

    def refresh_content(self):
        task_name = self.query_one("#sidebar").pressed_button.id_
        task = self.task_launcher.name2task[task_name]
        content = self.query_one("#content")
        content.clear()
        content.refresh()
        filt = "\x1b[H\x1b[2J\x1b[3J"
        log = task.buffer.decode(errors="replace").replace(filt, "\n")
        i = log.rfind(filt[0], -len(filt))
        if i != -1:
            log = log[:i]

        content.write(log)
        # content.refresh()

    def exit_when_finished(self):
        if all(x.finished for x in self.task_launcher.tasks) and all(
            not th.is_alive() for th in self.task_launcher.threads
        ):
            self.exit(
                "All tasks are finished!"
            )

    def on_mount(self) -> None:
        self.query_one("#sidebar").focus()
        self.refresh_timer = self.set_interval(0.9, self.refresh_content, pause=False)
        self.exit_timer = self.set_interval(10, self.exit_when_finished, pause=False)
        self.animation_timer = self.set_interval(
            2 / 10, self.refresh_animation, pause=False
        )

    def refresh_animation(self):
        for radio in self.query(RadioButton):
            task = self.task_launcher.name2task[radio.id_]
            if task.waiting:
                radio.label.plain = radio.id_ + " 💤"
                radio.label.style = "gray"
            elif not task.finished:
                idx = WORKING_ANIMATIONS.find(radio.label.plain[-1])
                symb = WORKING_ANIMATIONS[(idx + 1) % len(WORKING_ANIMATIONS)]
                radio.label.plain = radio.id_ + " " + symb
                if task.retry:
                    radio.label.style = "bold yellow"
                else:
                    radio.label.style = "bold"
            else:
                radio.label.plain = radio.id_ + " ✔"
                radio.label.style = "bold green"
            radio.refresh(layout=True)


if __name__ == "__main__":
    from wikidata_tools.build_wd.config import (
        OLD_WIKIDATA_DATE,
        NEW_WIKIDATA_DATE,
        STORAGE_FOLDER,
    )
    import os.path as osp
    import os
    import json

    os.makedirs(progress_folder, exist_ok=True)
    os.makedirs(osp.join(STORAGE_FOLDER, "errors"), exist_ok=True)
    d = {
        "old_date": OLD_WIKIDATA_DATE,
        "new_date": NEW_WIKIDATA_DATE,
    }
    with open(osp.join(STORAGE_FOLDER, "config.json"), "w") as f:
        json.dump(d, f)
    os.makedirs(osp.join(STORAGE_FOLDER, "errors"), exist_ok=True)
    task_launcher = TaskLauncher(tasks)
    task_launcher.run()
    app = DockLayoutExample(tasks)
    result = app.run()
    print(result)
