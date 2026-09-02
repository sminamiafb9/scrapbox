import gradio as gr


def greet(name):
    return "Hello " + name + "!"


def entry_point():
    demo = gr.Interface(fn=greet, inputs="text", outputs="text")
    demo.launch()


if __name__ == "__main__":
    entry_point()
