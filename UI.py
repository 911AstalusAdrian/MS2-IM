import customtkinter

from logic import run_main_logic


def encoder_option_callback(choice):
    print("Encoders dropdown clicked:", choice)


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("800x500")
        self.title("Modern vs Ancient DNA")

        # Options Frame
        self.options_frame = customtkinter.CTkFrame(self)
        self.label_desc = customtkinter.CTkLabel(self.options_frame, text="Please provide a DNA sequence and \npick the desired encoder and classification model",
                                                 fg_color="transparent")
        self.label_desc.grid(row=0, column=0, padx=15, pady=(10, 0), sticky="w")

        self.label_seq = customtkinter.CTkLabel(self.options_frame, text="DNA Sequence:", fg_color="transparent")
        self.label_seq.grid(row=1, column=0, padx=15, pady=(10, 0), sticky="w")

        self.sequence = customtkinter.CTkEntry(self.options_frame, placeholder_text="ACTCGCT...", width=300)
        self.sequence.grid(row=2, column=0, padx=10, pady=(10, 0), sticky="w")

        self.label_enc = customtkinter.CTkLabel(self.options_frame, text="Encoder:", fg_color="transparent")
        self.label_enc.grid(row=3, column=0, padx=15, pady=(10, 0), sticky="w")

        self.encoder_var = customtkinter.StringVar(value="OneHotEncoder")
        self.encoder_option = customtkinter.CTkOptionMenu(self.options_frame,
                                                          values=["OneHotEncoder", "KMerEncoder"],
                                                          command=encoder_option_callback,
                                                          variable=self.encoder_var,
                                                          width=300)
        self.encoder_option.grid(row=4, column=0, padx=10, pady=(10, 0), sticky="w")

        self.label_alg = customtkinter.CTkLabel(self.options_frame, text="Models:", fg_color="transparent")
        self.label_alg.grid(row=5, column=0, padx=15, pady=(10, 0), sticky="w")

        self.alg_var = customtkinter.StringVar(value="CNN Classifier")
        self.alg_option = customtkinter.CTkOptionMenu(self.options_frame, values=["CNN Classifier", "Random Forest Algorithm"],
                                                      command=self.alg_option_callback,
                                                      variable=self.alg_var, width=300)
        self.alg_option.grid(row=6, column=0, padx=10, pady=(10, 0), sticky="w")

        self.button_submit = customtkinter.CTkButton(self.options_frame, width=100, text="Submit",
                                                     command=self.submit_action)
        self.button_submit.grid(row=7, column=0, padx=100, pady=(25, 25), sticky="w")

        #Result
        self.result_frame = customtkinter.CTkFrame(self)
        self.label_result = customtkinter.CTkLabel(self.result_frame, text="Result:",
                                                      fg_color="transparent")
        self.label_result.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")

        # Stats Frame
        self.stats_frame = customtkinter.CTkFrame(self)
        self.label_desc_stats = customtkinter.CTkLabel(self.stats_frame, text="Evaluation Metrics           ", fg_color="transparent")
        self.label_desc_stats.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")

        self.accuracy_label = customtkinter.CTkLabel(self.stats_frame, text="Accuracy:", fg_color="transparent")
        self.accuracy_label.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="w")

        self.f1_label = customtkinter.CTkLabel(self.stats_frame, text="F1 Score:", fg_color="transparent")
        self.f1_label.grid(row=2, column=0, padx=10, pady=(10, 0), sticky="w")

        self.recall_label = customtkinter.CTkLabel(self.stats_frame, text="Recall:",
                                                        fg_color="transparent")
        self.recall_label.grid(row=3, column=0, padx=10, pady=(10, 0), sticky="w")

        # self.k_fold_label = customtkinter.CTkLabel(self.stats_frame, text="K-Fold: 0.83",
        #                                                 fg_color="transparent")
        # self.k_fold_label.grid(row=4, column=0, padx=10, pady=(10, 0), sticky="w")


        # Positioning
        self.options_frame.grid(row=0, column=0, rowspan=2, padx=20, pady=10, sticky="nsew")
        self.stats_frame.grid(row=1, column=1, padx=20, pady=10, sticky="nsew")
        self.result_frame.grid(row=0, column=1, padx=20, pady=10, sticky="nsew")

    def submit_action(self):
        user_input_sequence = self.sequence.get()
        encoder_choice = self.encoder_var.get()
        algorithm_choice = self.alg_var.get()

        # Get metrics and prediction
        result, accuracy, f1, recall = run_main_logic(user_input_sequence, encoder_choice, algorithm_choice)

        self.update_metrics(result, accuracy, f1, recall)

    def update_metrics(self, result, accuracy, f1, recall):
        self.accuracy_label.configure(text=f"Accuracy: {accuracy}")
        self.f1_label.configure(text=f"F1 Score: {f1}")
        self.recall_label.configure(text=f"Recall: {recall}")
        self.label_result.configure(text=(f"Result: {'Modern' if result else 'Ancient'}"))

    def alg_option_callback(self, choice):
        print("Algorithm dropdown clicked:", choice)


