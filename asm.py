import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import uuid

class TFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TFilter FIR Filter Design")
        self.root.geometry("1000x700")

        # Initialize variables
        self.fs = tk.DoubleVar(value=2000.0)  # Sampling rate in Hz
        self.fpass = tk.DoubleVar(value=400.0)  # Passband frequency in Hz
        self.fstop = tk.DoubleVar(value=500.0)  # Stopband frequency in Hz
        self.order = tk.IntVar(value=30)  # Filter order
        self.filter_type = tk.StringVar(value="lowpass")  # Filter type
        self.window_type = tk.StringVar(value="hamming")  # Window type
        self.taps = None
        self.filter_id = str(uuid.uuid4())[:8]  # Unique filter ID

        # Create GUI
        self.create_gui()

        # Initialize plot
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.plot_type = "frequency"  # Default plot type

        # Update plot with default values
        self.design_filter()

    def create_gui(self):
        # Input frame
        input_frame = ttk.LabelFrame(self.root, text="Filter Parameters", padding=10)
        input_frame.pack(padx=10, pady=10, fill=tk.X)

        # Sampling rate
        ttk.Label(input_frame, text="Sampling Rate (Hz):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(input_frame, textvariable=self.fs).grid(row=0, column=1, padx=5, pady=5)

        # Passband frequency
        ttk.Label(input_frame, text="Passband Frequency (Hz):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(input_frame, textvariable=self.fpass).grid(row=1, column=1, padx=5, pady=5)

        # Stopband frequency
        ttk.Label(input_frame, text="Stopband Frequency (Hz):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(input_frame, textvariable=self.fstop).grid(row=2, column=1, padx=5, pady=5)

        # Filter order
        ttk.Label(input_frame, text="Filter Order:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(input_frame, textvariable=self.order).grid(row=3, column=1, padx=5, pady=5)

        # Filter type
        ttk.Label(input_frame, text="Filter Type:").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        filter_types = ["lowpass", "highpass", "bandpass", "bandstop"]
        ttk.Combobox(input_frame, textvariable=self.filter_type, values=filter_types, state="readonly").grid(row=4, column=1, padx=5, pady=5)

        # Window type
        ttk.Label(input_frame, text="Window Type:").grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        window_types = ["hamming", "hann", "blackman", "kaiser"]
        ttk.Combobox(input_frame, textvariable=self.window_type, values=window_types, state="readonly").grid(row=5, column=1, padx=5, pady=5)

        # Buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Design Filter", command=self.design_filter).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Show Frequency Response", command=lambda: self.plot_response("frequency")).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Show Impulse Response", command=lambda: self.plot_response("impulse")).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Generate C++ Code", command=self.generate_cpp_code).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Generate Assembly Code", command=self.generate_asm_code).pack(side=tk.LEFT, padx=5)

        # Coefficients display
        coeff_frame = ttk.LabelFrame(self.root, text="Filter Coefficients", padding=10)
        coeff_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.coeff_text = tk.Text(coeff_frame, height=5, width=50)
        self.coeff_text.pack(fill=tk.BOTH, expand=True)
        self.coeff_text.insert(tk.END, "Coefficients will appear here after designing the filter.")
        self.coeff_text.config(state=tk.DISABLED)

        # Plot frame
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    def design_filter(self):
        try:
            fs = self.fs.get()
            fpass = self.fpass.get()
            fstop = self.fstop.get()
            order = self.order.get()
            filter_type = self.filter_type.get()
            window_type = self.window_type.get()

            # Validate inputs
            if fs <= 0 or fpass <= 0 or fstop <= 0 or order <= 0:
                messagebox.showerror("Error", "All numerical inputs must be positive.")
                return
            if fpass >= fs / 2 or fstop >= fs / 2:
                messagebox.showerror("Error", "Frequencies must be less than Nyquist frequency (fs/2).")
                return
            if filter_type in ["bandpass", "bandstop"] and fstop <= fpass:
                messagebox.showerror("Error", "Stopband frequency must be greater than passband for bandpass/bandstop.")
                return

            # Design filter
            if filter_type == "lowpass":
                self.taps = signal.firwin(order + 1, fpass / (fs / 2), window=window_type, pass_zero=True)
            elif filter_type == "highpass":
                self.taps = signal.firwin(order + 1, fpass / (fs / 2), window=window_type, pass_zero=False)
            elif filter_type == "bandpass":
                self.taps = signal.firwin(order + 1, [fpass / (fs / 2), fstop / (fs / 2)], window=window_type, pass_zero=False)
            elif filter_type == "bandstop":
                self.taps = signal.firwin(order + 1, [fpass / (fs / 2), fstop / (fs / 2)], window=window_type, pass_zero=True)

            # Update coefficients display
            self.coeff_text.config(state=tk.NORMAL)
            self.coeff_text.delete(1.0, tk.END)
            self.coeff_text.insert(tk.END, f"Filter ID: {self.filter_id}\n\n")
            self.coeff_text.insert(tk.END, "Coefficients:\n")
            for i, tap in enumerate(self.taps):
                self.coeff_text.insert(tk.END, f"h[{i}] = {tap:.15f};\n")
            self.coeff_text.config(state=tk.DISABLED)

            # Update plot
            self.plot_response(self.plot_type)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to design filter: {str(e)}")

    def plot_response(self, plot_type):
        if self.taps is None:
            messagebox.showwarning("Warning", "Design a filter first.")
            return

        self.plot_type = plot_type
        self.ax1.clear()
        self.ax2.clear()

        fs = self.fs.get()
        if plot_type == "frequency":
            w, h = signal.freqz(self.taps, worN=8000)
            freq = w * fs / (2 * np.pi)
            self.ax1.plot(freq, 20 * np.log10(np.abs(h)))
            self.ax1.set_title("Frequency Response")
            self.ax1.set_xlabel("Frequency (Hz)")
            self.ax1.set_ylabel("Magnitude (dB)")
            self.ax1.grid(True)

            self.ax2.plot(freq, np.unwrap(np.angle(h)) * 180 / np.pi)
            self.ax2.set_xlabel("Frequency (Hz)")
            self.ax2.set_ylabel("Phase (degrees)")
            self.ax2.grid(True)
        else:  # Impulse response
            self.ax1.stem(self.taps, use_line_collection=True)
            self.ax1.set_title("Impulse Response")
            self.ax1.set_xlabel("Sample")
            self.ax1.set_ylabel("Amplitude")
            self.ax1.grid(True)

            self.ax2.axis("off")  # No phase plot for impulse response

        self.fig.tight_layout()
        self.canvas.draw()

    def generate_cpp_code(self):
        if self.taps is None:
            messagebox.showwarning("Warning", "Design a filter first.")
            return

        # Generate C++ code
        code = f"""// FIR Filter Implementation
// Filter ID: {self.filter_id}
// Filter Type: {self.filter_type.get()}
// Sampling Rate: {self.fs.get()} Hz
// Passband Frequency: {self.fpass.get()} Hz
// Stopband Frequency: {self.fstop.get()} Hz
// Filter Order: {self.order.get()}

#include <vector>

class FIRFilter {{
private:
    std::vector<double> coefficients;
    std::vector<double> buffer;
    size_t index;

public:
    FIRFilter() : index(0) {{
        coefficients = {{
            {", ".join([f"{tap:.15f}" for tap in self.taps])}
        }};
        buffer.resize(coefficients.size(), 0.0);
    }}

    double filter(double input) {{
        buffer[index] = input;
        double output = 0.0;
        for (size_t i = 0; i < coefficients.size(); ++i) {{
            size_t idx = (index + coefficients.size() - i) % coefficients.size();
            output += coefficients[i] * buffer[idx];
        }}
        index = (index + 1) % coefficients.size();
        return output;
    }}
}};
"""
        # Show C++ code in a new window
        cpp_window = tk.Toplevel(self.root)
        cpp_window.title("Generated C++ Code")
        cpp_window.geometry("600x400")
        text_area = tk.Text(cpp_window, wrap=tk.WORD)
        text_area.pack(fill=tk.BOTH, expand=True)
        text_area.insert(tk.END, code)
        text_area.config(state=tk.DISABLED)

    def generate_asm_code(self):
        if self.taps is None:
            messagebox.showwarning("Warning", "Design a filter first.")
            return

        # Generate x86-64 Assembly code
        code = f"""; FIR Filter Implementation (x86-64)
; Filter ID: {self.filter_id}
; Filter Type: {self.filter_type.get()}
; Sampling Rate: {self.fs.get()} Hz
; Passband Frequency: {self.fpass.get()} Hz
; Stopband Frequency: {self.fstop.get()} Hz
; Filter Order: {self.order.get()}

section .data
    ; Filter coefficients (double precision floating-point)
    coefficients dq {", ".join([f"{tap:.15f}" for tap in self.taps])}
    coeff_size equ {len(self.taps)}
    buffer times {len(self.taps)} dq 0.0
    index dq 0

section .text
global fir_filter
fir_filter:
    ; Input: xmm0 = input sample (double)
    ; Output: xmm0 = output sample (double)
    ; Clobbers: rax, rcx, rdx, xmm1, xmm2

    ; Save input
    movsd [buffer + index * 8], xmm0

    ; Initialize output
    xorpd xmm0, xmm0

    ; Loop counter
    mov rcx, coeff_size
    mov rax, index
    mov rdx, coeff_size

.loop:
    ; Get buffer index
    mov rbx, rax
    sub rbx, rcx
    add rbx, rdx
    xor rdx, rdx
    div rbx, coeff_size
    mov rbx, rdx

    ; Multiply coefficient with buffer value
    movsd xmm1, [coefficients + rcx * 8 - 8]
    movsd xmm2, [buffer + rbx * 8]
    mulsd xmm1, xmm2
    addsd xmm0, xmm1

    loop .loop

    ; Update index
    inc qword [index]
    mov rax, [index]
    cmp rax, coeff_size
    jb .done
    xor rax, rax
    mov [index], rax

.done:
    ret
"""
        # Show Assembly code in a new window
        asm_window = tk.Toplevel(self.root)
        asm_window.title("Generated Assembly Code")
        asm_window.geometry("600x400")
        text_area = tk.Text(asm_window, wrap=tk.WORD)
        text_area.pack(fill=tk.BOTH, expand=True)
        text_area.insert(tk.END, code)
        text_area.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = TFilterApp(root)
    root.mainloop()