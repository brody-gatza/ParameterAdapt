
import customtkinter
import ui_solver_bridge

class UI(customtkinter.CTk):

    def __init__(self):

        super().__init__()

        customtkinter.set_appearance_mode("Dark")      # Modes: system (default), light, dark
        customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

        self.window_width = 1360
        self.window_length= 720

        self.screen_width  = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()

        self.x = (self.screen_width // 2) - (self.window_width // 2)
        self.y = (self.screen_height // 2) - (self.window_length // 2)

        self.geometry('{}x{}+{}+{}'.format(self.window_width, self.window_length, self.x, self.y))
        self.title('ROMify')
       
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1,weight=1)

        ######################   Navigation Frame   ######################

        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)

        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="ROMify", compound="center" , font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=0, pady=20)

        self.home_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Home",fg_color="transparent",  text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),anchor="w",command=self.home_button_callback)
        self.home_button.grid(row=1, column=0, sticky="ew")

        self.set_up_page_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Setup", fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), anchor="w",command=self.setup_button_callback)
        self.set_up_page_button.grid(row=2, column=0, sticky="ew")

        self.results_page_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Results", fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), anchor="w",command=self.results_button_callback)
        self.results_page_button.grid(row=3, column=0, sticky="ew")

        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.home_frame.grid_columnconfigure(1, weight=1)

        self.set_up_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.set_up_frame.grid_columnconfigure(4,weight=1)

        self.results_page_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.results_page_frame.grid_columnconfigure(1, weight=1)

        ######################   Time Descrtization   ######################

        self.time_descritization_frame = customtkinter.CTkFrame(self.set_up_frame,corner_radius=20)
        self.time_descritization_frame.grid(row=0,column=0,padx=10,pady=10,sticky='new')

        self.time_descritization_group_label = customtkinter.CTkLabel(self.time_descritization_frame,text="Time Discretization" )
        self.time_descritization_group_label.grid(row=0 , column= 0 , columnspan = 2 )

        self.dt_label = customtkinter.CTkLabel(self.time_descritization_frame,text='Time Step')
        self.dt_label.grid(row=1,column=0,padx=15,pady=10)

        self.dt_entry_var = customtkinter.StringVar()
        self.dt = customtkinter.CTkEntry(self.time_descritization_frame , textvariable=self.dt_entry_var )
        self.dt.grid(row=1,column=1,padx=15,pady=10)

        self.num_step_label = customtkinter.CTkLabel(self.time_descritization_frame,text='Number of Steps')
        self.num_step_label.grid(row=2,column=0,padx=15,pady=10)

        self.num_step_entry_var = customtkinter.StringVar()
        self.num_step = customtkinter.CTkEntry(self.time_descritization_frame,textvariable=self.num_step_entry_var)
        self.num_step.grid(row=2,column=1,padx=15,pady=10)

        self.time_scheme_label = customtkinter.CTkLabel(self.time_descritization_frame,text='Time Scheme')
        self.time_scheme_label.grid(row=3,column=0,padx=15,pady=10)

        self.time_scheme_entry_var = customtkinter.StringVar()
        self.time_scheme_options = ['Explicit - RK4' , 'Implicit - BDF']
        self.time_scheme = customtkinter.CTkOptionMenu(self.time_descritization_frame , values=self.time_scheme_options , 
                                                       variable=self.time_scheme_entry_var , command = self.time_scheme_callback)
        self.time_scheme.grid(row=3,column=1,padx=15,pady=10)

        self.dual_time_label = customtkinter.CTkLabel(self.time_descritization_frame,text='Dual Time')
        self.dual_time_label.grid(row=4,column=0,padx=15,pady=10)

        self.dual_time_entry_var = customtkinter.StringVar()
        self.dual_time_options = ['Yes' , 'No']
        self.dual_time = customtkinter.CTkOptionMenu(self.time_descritization_frame , values=self.dual_time_options ,variable=self.dual_time_entry_var)
        self.dual_time.grid(row=4,column=1,padx=15,pady=10)

        self.res_tol_label = customtkinter.CTkLabel(self.time_descritization_frame,text='Residual Tol.')
        self.res_tol_label.grid(row=5,column=0,padx=15,pady=10)

        self.res_tol_entry_var = customtkinter.StringVar()
        self.res_tol = customtkinter.CTkEntry(self.time_descritization_frame,textvariable=self.res_tol_entry_var)
        self.res_tol.grid(row=5,column=1,padx=15,pady=10)

        ######################   Space Descrtization   ######################

        self.space_descritization_frame = customtkinter.CTkFrame(self.set_up_frame,corner_radius=20)
        self.space_descritization_frame.grid(row=1,column=0,padx=10,pady=10 , sticky = 'new')

        self.space_descritization_group_label = customtkinter.CTkLabel(self.space_descritization_frame,text="Space Discretization")
        self.space_descritization_group_label.grid(row=0 , column= 0 , columnspan = 2) 

        
        self.mesh_x_init_label = customtkinter.CTkLabel(self.space_descritization_frame , text='x initial')
        self.mesh_x_init_label.grid(row=1,column=0,padx=25, pady=10)
        
        self.mesh_x_init_entry_var = customtkinter.StringVar()
        self.mesh_x_init = customtkinter.CTkEntry(self.space_descritization_frame , textvariable=self.mesh_x_init_entry_var)
        self.mesh_x_init.grid(row=1,column=1,padx=25, pady=10)

        self.mesh_x_final_label = customtkinter.CTkLabel(self.space_descritization_frame , text='x final')
        self.mesh_x_final_label.grid(row=2,column=0,padx=25, pady=10)

        self.mesh_x_final_entry_var = customtkinter.StringVar()
        self.mesh_x_final= customtkinter.CTkEntry(self.space_descritization_frame , textvariable=self.mesh_x_final_entry_var)
        self.mesh_x_final.grid(row=2,column=1,padx=25, pady=10)

        self.mesh_cell_number_label = customtkinter.CTkLabel(self.space_descritization_frame , text='cell number')
        self.mesh_cell_number_label.grid(row=3,column=0,padx=25, pady=10)

        self.cell_number_entry_var = customtkinter.StringVar()
        self.cell_number = customtkinter.CTkEntry(self.space_descritization_frame , textvariable=self.cell_number_entry_var)
        self.cell_number.grid(row=3,column=1,padx=25, pady=10)

        ######################   Inet BC   ######################

        self.inlet_bc_frame = customtkinter.CTkFrame(self.set_up_frame,corner_radius=20)
        self.inlet_bc_frame.grid(row=0,column=1,padx=10,pady=10 , sticky = 'new')

        self.inlet_label = customtkinter.CTkLabel(self.inlet_bc_frame,text='Inlet Boundary Condition')
        self.inlet_label.grid(row=0,column=0,columnspan = 2)

        self.inlet_press_label = customtkinter.CTkLabel(self.inlet_bc_frame,text='Pressure')
        self.inlet_press_label.grid(row=1,column=0,padx=15,pady=10) 

        self.inlet_press_entry_var = customtkinter.StringVar()
        self.inlet_press = customtkinter.CTkEntry(self.inlet_bc_frame , textvariable=self.inlet_press_entry_var)
        self.inlet_press.grid(row=1,column=1,padx=15,pady=10) 

        self.inlet_temp_label = customtkinter.CTkLabel(self.inlet_bc_frame,text='Temperature')
        self.inlet_temp_label.grid(row=2,column=0,padx=15,pady=10)  

        self.inlet_temp_entry_var = customtkinter.StringVar()
        self.inlet_temp = customtkinter.CTkEntry(self.inlet_bc_frame,textvariable=self.inlet_temp_entry_var)
        self.inlet_temp.grid(row=2,column=1,padx=15,pady=10) 

        self.inlet_vel_label = customtkinter.CTkLabel(self.inlet_bc_frame,text='Velocity')
        self.inlet_vel_label.grid(row=3,column=0,padx=15,pady=10) 

        self.inlet_vel_entry_var = customtkinter.StringVar()
        self.inlet_vel = customtkinter.CTkEntry(self.inlet_bc_frame , textvariable=self.inlet_vel_entry_var)
        self.inlet_vel.grid(row=3,column=1,padx=15,pady=10) 

        self.inlet_rho_label = customtkinter.CTkLabel(self.inlet_bc_frame,text='Density')
        self.inlet_rho_label.grid(row=4,column=0,padx=15,pady=10) 

        self.inlet_rho_entry_var = customtkinter.StringVar()
        self.inlet_rho = customtkinter.CTkEntry(self.inlet_bc_frame,textvariable=self.inlet_rho_entry_var)
        self.inlet_rho.grid(row=4,column=1,padx=15,pady=10) 

        self.inlet_mass_frac_label = customtkinter.CTkLabel(self.inlet_bc_frame,text='Mass Fraction')
        self.inlet_mass_frac_label.grid(row=5,column=0,padx=15,pady=10) 

        self.inlet_mass_frac_entry_var = customtkinter.StringVar()
        self.inlet_mass_frac = customtkinter.CTkEntry(self.inlet_bc_frame,textvariable=self.inlet_mass_frac_entry_var)
        self.inlet_mass_frac.grid(row=5,column=1,padx=15,pady=10) 

        ######################   Outlet BC   ######################

        self.outlet_bc_frame = customtkinter.CTkFrame(self.set_up_frame,corner_radius=20)
        self.outlet_bc_frame.grid(row=0,column=2,padx=10,pady=10 , sticky = 'new')

        self.outlet_label = customtkinter.CTkLabel(self.outlet_bc_frame,text='Outlet Boundary Condition')
        self.outlet_label.grid(row=0,column=0,columnspan = 2)

        self.outlet_press_label = customtkinter.CTkLabel(self.outlet_bc_frame,text='Pressure')
        self.outlet_press_label.grid(row=1,column=0,padx=15,pady=10) 

        self.outlet_press_entry_var = customtkinter.StringVar()
        self.outlet_press = customtkinter.CTkEntry(self.outlet_bc_frame,textvariable=self.outlet_press_entry_var)
        self.outlet_press.grid(row=1,column=1,padx=15,pady=10) 

        self.outlet_temp_label = customtkinter.CTkLabel(self.outlet_bc_frame,text='Temperature')
        self.outlet_temp_label.grid(row=2,column=0,padx=15,pady=10)  

        self.outlet_temp_entry_var = customtkinter.StringVar()
        self.outlet_temp = customtkinter.CTkEntry(self.outlet_bc_frame,textvariable=self.outlet_temp_entry_var)
        self.outlet_temp.grid(row=2,column=1,padx=15,pady=10) 

        self.outlet_vel_label = customtkinter.CTkLabel(self.outlet_bc_frame,text='Velocity')
        self.outlet_vel_label.grid(row=3,column=0,padx=15,pady=10) 

        self.outlet_vel_entry_var = customtkinter.StringVar()
        self.outlet_vel = customtkinter.CTkEntry(self.outlet_bc_frame,textvariable=self.outlet_vel_entry_var)
        self.outlet_vel.grid(row=3,column=1,padx=15,pady=10) 

        self.outlet_rho_label = customtkinter.CTkLabel(self.outlet_bc_frame,text='Density')
        self.outlet_rho_label.grid(row=4,column=0,padx=15,pady=10) 

        self.outlet_rho_entry_var = customtkinter.StringVar()
        self.outlet_rho = customtkinter.CTkEntry(self.outlet_bc_frame,textvariable=self.outlet_rho_entry_var)
        self.outlet_rho.grid(row=4,column=1,padx=15,pady=10) 

        self.outlet_mass_frac_label = customtkinter.CTkLabel(self.outlet_bc_frame,text='Mass Fraction')
        self.outlet_mass_frac_label.grid(row=5,column=0,padx=15,pady=10) 

        self.outlet_mass_frac_entry_var = customtkinter.StringVar()
        self.outlet_mass_frac = customtkinter.CTkEntry(self.outlet_bc_frame,textvariable=self.outlet_mass_frac_entry_var)
        self.outlet_mass_frac.grid(row=5,column=1,padx=15,pady=10)

        ######################   Gas Model   ######################
        
        self.misc_frame = customtkinter.CTkFrame(self.set_up_frame,corner_radius=20)
        self.misc_frame.grid(row=1,column=1,padx=10,pady=10 , sticky = 'new')

        self.misc_label = customtkinter.CTkLabel(self.misc_frame,text="Gas Model")
        self.misc_label.grid(row=0 , column= 0 , columnspan = 2) 

        self.gas_model_label = customtkinter.CTkLabel(self.misc_frame , text='Gas Model')
        self.gas_model_label.grid(row=1,column=0,padx=25, pady=10)

        self.gas_model_entry_var = customtkinter.StringVar()
        self.gas_model_options = ['Ideal Gas Model']
        self.gas_model = customtkinter.CTkOptionMenu(self.misc_frame , values=self.gas_model_options)
        self.gas_model.grid(row=1,column=1,padx=15, pady=10)
        
        ######################   ROM   ######################

        self.rom_frame = customtkinter.CTkFrame(self.set_up_frame,corner_radius=20)
        self.rom_frame.grid(row=1,column=2,padx=10,pady=10 , sticky = 'new')

        self.misc_label = customtkinter.CTkLabel(self.rom_frame,text="Reduced Order Modeling (ROM)")
        self.misc_label.grid(row=0 , column= 0 , columnspan = 2)

        self.rom_method_checkbox_check_var = customtkinter.BooleanVar()
        self.rom_method_checkbox = customtkinter.CTkCheckBox(self.rom_frame , text='Enable ROM', variable=self.rom_method_checkbox_check_var ,command=self.ROM_checkbox_callback)
        self.rom_method_checkbox.grid(row=1,column=0,padx=5, pady=10)
        
        self.rom_method_entry_var = customtkinter.StringVar()
        self.rom_method_options = ['Galerking','LSPG']
        self.rom_method = customtkinter.CTkOptionMenu(self.rom_frame , values=self.rom_method_options , state='disabled')
        self.rom_method.grid(row=1,column=1,padx=5, pady=10)

        self.energy_capture_label = customtkinter.CTkLabel(self.rom_frame , text='POD Energy Capture')
        self.energy_capture_label.grid(row=3,column=0,padx=5, pady=10)
        
        self.energy_capture_entry_var = customtkinter.StringVar()
        self.energy_capture = customtkinter.CTkEntry(self.rom_frame , textvariable=self.energy_capture_entry_var , state='disabled')
        self.energy_capture.grid(row=3,column=1,padx=5, pady=10)

        self.hyper_method_checkbox_check_var = customtkinter.BooleanVar()
        self.hyper_method_checkbox = customtkinter.CTkCheckBox(self.rom_frame , text='Hyper-Reduction', variable=self.hyper_method_checkbox_check_var , command=self.hyper_checkbox_callback)
        self.hyper_method_checkbox.grid(row=4,column=0,padx=5, pady=10)

        self.hyper_method_entry_var = customtkinter.StringVar()
        self.hyper_method_options = ['Gappy POD']
        self.hyper_method = customtkinter.CTkOptionMenu(self.rom_frame , values=self.hyper_method_options ,state='disabled')
        self.hyper_method.grid(row=4,column=1,padx=5, pady=10)        

        ######################   Input File  ######################

        self.input_file_frame = customtkinter.CTkFrame(self.set_up_frame,corner_radius=20)
        self.input_file_frame.grid_columnconfigure(0,weight=1)
        self.input_file_frame.grid_columnconfigure(1,weight=5)
        self.input_file_frame.grid_columnconfigure(2,weight=1)
        self.input_file_frame.grid(row=2,column=0,columnspan=3,pady=10 , sticky = 'new')
        
        self.input_file_label = customtkinter.CTkLabel(self.input_file_frame , text='Input Files')
        self.input_file_label.grid(row=0,column=0,pady=10)   

        self.input_file = customtkinter.CTkEntry(self.input_file_frame , placeholder_text= 'Enter the Input File Path')
        self.input_file.grid(row=0,column=1,pady=10,sticky='ew')    

        self.FOM_file_label = customtkinter.CTkLabel(self.input_file_frame , text='FOM Result')
        self.FOM_file_label.grid(row=1,column=0,pady=10)  

        self.FOM_file = customtkinter.CTkEntry(self.input_file_frame , placeholder_text= 'Enter the FOM Result Path')
        self.FOM_file.grid(row=1,column=1,pady=10,sticky='ew')

        self.input_file_button = customtkinter.CTkButton(self.input_file_frame , text='Open' , command=self.input_file_open_callback)
        self.input_file_button.grid(row=0,column=2)

        self.FOM_file_button = customtkinter.CTkButton(self.input_file_frame , text='Open' , command=self.FOM_file_open_callback)
        self.FOM_file_button.grid(row=1,column=2)

        ######################  Visualization  ######################

        self.visual_frame = customtkinter.CTkFrame(self.set_up_frame,corner_radius=20)
        self.visual_frame.grid(row=0,column=3,padx=10,pady=10 , sticky = 'new')

        self.visual_label = customtkinter.CTkLabel(self.visual_frame , text='Visualization')
        self.visual_label.grid(row=0,column=0,pady=10,columnspan=2)

        self.visual_options = ['None','Density','Pressure','Temprature','Velocity','Heat Release','Mass Fraction']

        self.visual_1_option_label = customtkinter.CTkLabel(self.visual_frame , text='Variable #1')
        self.visual_1_option_label.grid(row=1,column=0,padx=15,pady=10)

        self.visual_1_option_entry_var=customtkinter.StringVar(value=self.visual_options[0])
        self.visual_1_option = customtkinter.CTkOptionMenu(self.visual_frame , values=self.visual_options , variable=self.visual_1_option_entry_var)
        self.visual_1_option.grid(row=1,column=1,padx=5, pady=10)

        self.visual_2_option_label = customtkinter.CTkLabel(self.visual_frame , text='Variable #2')
        self.visual_2_option_label.grid(row=2,column=0,padx=15,pady=10)

        self.visual_2_option_entry_var=customtkinter.StringVar(value=self.visual_options[0])
        self.visual_2_option = customtkinter.CTkOptionMenu(self.visual_frame , values=self.visual_options , variable=self.visual_2_option_entry_var)
        self.visual_2_option.grid(row=2,column=1,padx=5, pady=10)

        self.visual_3_option_label = customtkinter.CTkLabel(self.visual_frame , text='Variable #3')
        self.visual_3_option_label.grid(row=3,column=0,padx=15,pady=10)

        self.visual_3_option_entry_var=customtkinter.StringVar(value=self.visual_options[0])
        self.visual_3_option = customtkinter.CTkOptionMenu(self.visual_frame , values=self.visual_options , variable=self.visual_3_option_entry_var)
        self.visual_3_option.grid(row=3,column=1,padx=5, pady=10)

        self.visual_4_option_label = customtkinter.CTkLabel(self.visual_frame , text='Variable #4')
        self.visual_4_option_label.grid(row=4,column=0,padx=15,pady=10)

        self.visual_4_option_entry_var=customtkinter.StringVar(value=self.visual_options[0])
        self.visual_4_option = customtkinter.CTkOptionMenu(self.visual_frame , values=self.visual_options , variable=self.visual_4_option_entry_var)
        self.visual_4_option.grid(row=4,column=1,padx=5, pady=10)                

        self.run_button = customtkinter.CTkButton(self.set_up_frame , text='RUN !',font=customtkinter.CTkFont(size=30),command=self.run_callback)
        self.run_button.grid(row=3,column=1,columnspan=2,pady=20,sticky='news')

        # self.home_button_callback()
        self.setup_button_callback()

    def select_frame_by_name(self, name):

        self.home_button.configure(fg_color=("gray75", "gray25") if name == "Home" else "transparent")
        self.set_up_page_button.configure(fg_color=("gray75", "gray25") if name == "Setup" else "transparent")
        self.results_page_button.configure(fg_color=("gray75", "gray25") if name == "Results" else "transparent")

        if name == "Home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.home_frame.grid_forget()
        if name == "Setup":
            self.set_up_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.set_up_frame.grid_forget()
        if name == "Results":
            self.results_page_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.results_page_frame.grid_forget()

    def home_button_callback(self):

        self.select_frame_by_name("Home")
        self.description_textbox = customtkinter.CTkTextbox(master=self.home_frame , font=customtkinter.CTkFont(size=15, weight="bold"))
        self.description_textbox.grid(row=0, column=1 ,padx = 20 , pady =20 ,sticky="nsew")
        self.description_textbox.insert("0.0", "ROMify is a 1D compressible reacting flow solver with reduced-order-model (ROM) features. It aims to provide a simple and easy-to-use testbed to quickly prototype and examine new data-driven methods")

    def setup_button_callback(self):

        self.select_frame_by_name("Setup")
        
    def results_button_callback(self):

        self.select_frame_by_name("Results")

    def time_scheme_callback(self,time_scheme_entry_var):

        selected_option = time_scheme_entry_var

        if selected_option == 'Explicit - RK4':

            self.res_tol.configure(state='disabled')

        elif selected_option == 'Implicit - BDF':

            self.res_tol.configure(state='normal')

        
    def ROM_checkbox_callback(self):

        if self.rom_method_checkbox_check_var.get() == True :

            self.rom_method.configure(state = 'normal')
            self.energy_capture.configure(state = 'normal')
            self.FOM_file.configure(state = 'normal')
            self.FOM_file_button.configure(state = 'normal')

        else: 

            self.rom_method.configure(state = 'disabled')
            self.energy_capture.configure(state = 'disabled')
            self.FOM_file.configure(state = 'disabled')
            self.FOM_file_button.configure(state = 'disabled')


    def hyper_checkbox_callback(self):

        if self.hyper_method_checkbox_check_var.get() == True :

            self.hyper_method.configure(state = 'normal')

        else: 

            self.hyper_method.configure(state = 'disabled')    
            

    def input_file_open_callback(self):

        self.input_file_path = customtkinter.filedialog.askopenfilename()
        self.input_file.delete(0, customtkinter.END)
        self.input_file.insert(0,string=self.input_file_path)

        try:
            with open(self.input_file_path, 'r') as input_file:
                content = input_file.readlines()

            self.parameters = {}

            for line in content:
                line = line.strip()

                if not line or line.startswith('#'):
                    continue
                keyword_value = line.split('=',1)
                if len(keyword_value)==2:
                    keyword = keyword_value[0].strip()
                    value=keyword_value[1].strip()
                    self.parameters[keyword] = value

            self.fill_gui_from_inp_file()
            
        except FileNotFoundError:

            print(f"Error: File not found.")
            return None
    
        except IOError:

            print(f"Error: Unable to read the file.")
            return None
        
        
    def FOM_file_open_callback(self):

        self.FOM_file_path = customtkinter.filedialog.askopenfilename()
        self.FOM_file.delete(0, customtkinter.END)
        self.FOM_file.insert(0,string=self.FOM_file_path)


    def fill_gui_from_inp_file(self):

        parameters = self.parameters
        visual_options = self.visual_options

        for variable in parameters:

            if variable == 'dt':

                self.dt_entry_var.set(parameters['dt']) 

            if variable == 'num_steps':

                self.num_step_entry_var.set(parameters['num_steps'])

            if variable == 'time_scheme':

                if parameters['time_scheme'] == 'explicit_rk4':

                    self.time_scheme_entry_var.set(self.time_scheme_options[0])
                
                elif parameters['time_scheme'] == 'implicit_bdf':

                    self.time_scheme_entry_var.set(self.time_scheme_options[1])

            if variable == 'dual_time':

                if parameters['dual_time'] == 'true':

                    self.dual_time_entry_var.set(self.dual_time_options[0])
                
                elif parameters['dual_time'] == 'false':

                    self.dual_time_entry_var.set(self.dual_time_options[1])

            if variable == 'res_tol':

                self.res_tol_entry_var.set(parameters['res_tol'])

            if variable == 'x_initial':

                self.mesh_x_init_entry_var.set(parameters['x_initial'])

            if variable == 'x_final':

                self.mesh_x_final_entry_var.set(parameters['x_final'])
            
            if variable == 'cell_number':

                self.cell_number_entry_var.set(parameters['cell_number'])

            if variable == 'press_inlet':

                self.inlet_press_entry_var.set(parameters['press_inlet'])
            
            if variable == 'temp_inlet':

                self.inlet_temp_entry_var.set(parameters['temp_inlet'])

            if variable == 'vel_inlet':

                self.inlet_vel_entry_var.set(parameters['vel_inlet'])

            if variable == 'rho_inlet':

                self.inlet_rho_entry_var.set(parameters['rho_inlet'])                       

            if variable == 'mass_frac_inlet':

                self.inlet_mass_frac_entry_var.set(parameters['mass_frac_inlet'])    

            if variable == 'press_outlet':

                self.outlet_press_entry_var.set(parameters['press_outlet'])
            
            if variable == 'temp_outlet':

                self.outlet_temp_entry_var.set(parameters['temp_outlet'])

            if variable == 'vel_outlet':

                self.outlet_vel_entry_var.set(parameters['vel_outlet'])

            if variable == 'rho_outlet':

                self.outlet_rho_entry_var.set(parameters['rho_outlet'])                  

            if variable == 'mass_frac_outlet':

                self.outlet_mass_frac_entry_var.set(parameters['mass_frac_outlet'])

            if variable == 'gas_model':

                if parameters['gas_model'] == 'ideal gas model':

                    self.gas_model_entry_var.set(self.gas_model_options[0])

            if variable == 'calc_rom':

                if parameters['calc_rom'] == 'false':

                    self.rom_method_checkbox_check_var.set(False)

                elif parameters['calc_rom'] == 'true':

                    self.rom_method_checkbox_check_var.set(True)

            if variable == 'rom_method':

                if parameters['rom_method'] == 'galerkin':

                    self.rom_method_entry_var.set(self.rom_method_options[0])
                
                elif parameters['rom_method'] == 'lspg':

                    self.rom_method_entry_var.set(self.rom_method_options[1])

                self.ROM_checkbox_callback()

            if variable == 'pod_energy':

                self.energy_capture_entry_var.set(parameters['pod_energy'])

            if variable == 'hyper':

                if parameters['hyper'] == 'false':

                    self.hyper_method_checkbox_check_var.set(False)

                elif parameters['hyper'] == 'true':
                    
                    self.hyper_method_checkbox_check_var.set(True)

            if variable == 'hyper_method':

                if parameters['hyper_method'] == 'gappy_pod':

                    self.hyper_method_entry_var.set(self.hyper_method_options[0])

                self.hyper_checkbox_callback()

            if variable == 'variable1':
                
                indices = [index for index, value in enumerate(visual_options) if value == parameters['variable1']]
                
                self.visual_1_option_entry_var.set(visual_options[indices[0]])

            if variable == 'variable2':
                
                indices = [index for index, value in enumerate(visual_options) if value == parameters['variable2']]
                
                self.visual_2_option_entry_var.set(visual_options[indices[0]])

            if variable == 'variable3':
                
                indices = [index for index, value in enumerate(visual_options) if value == parameters['variable3']]
                
                self.visual_3_option_entry_var.set(visual_options[indices[0]])

            if variable == 'variable4':
                
                indices = [index for index, value in enumerate(visual_options) if value == parameters['variable4']]
                
                self.visual_4_option_entry_var.set(visual_options[indices[0]])      

    def run_callback(self):

        ui_solver_bridge.driver(self)
                                      
if __name__ == "__main__":

    app = UI()
    app.mainloop()