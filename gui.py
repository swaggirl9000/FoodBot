from tkinter import *
from chat import response, bot_name

BG_GRAY = "#f282b4"
BG_COLOR = "#fde4e3"
TEXT_COLOR = "#ef415e"

FONT = "Georgia 14"
BOLD = "Georgia 14 bold"

class ChatApp:
    
    def __init__(self):
        self.window = Tk()
        self.main_window()
        
    def run(self):
        self.window.mainloop()

    def main_window(self):
        self.window.title("Chat")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=500, height=550, bg=BG_COLOR)

        lab = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR, text = "What do you wanna eat?", font=BOLD, pady=10)
        lab.place(relwidth=1)

        line = Label(self.window, width=480, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        self.text_widget = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)

        bottom = Label(self.window, bg=BG_GRAY, height=80)
        bottom.place(relwidth=1, rely=0.825)

        self.message = Entry(bottom, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT)
        self.message.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.message.focus()
        self.message.bind("<Return>", self.pressed)

        send= Button(bottom, text="Send", font=BOLD, width=20, bg=BG_GRAY, command=lambda: self.pressed(None))
        send.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    def pressed(self, event):
        msg = self.message.get()
        self.insert(msg, "You")

    def insert(self,message, sender):
        if not message:
            return

        self.message.delete(0, END)
        msg = f"{sender}: {message}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg)
        self.text_widget.configure(state=DISABLED)

        msg2 = f"{bot_name}: {response(message)}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)
        
        self.text_widget.see(END)

if __name__ == "__main__":
    app = ChatApp()
    app.run()