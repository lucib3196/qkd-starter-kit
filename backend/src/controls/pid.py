class PIDController:
    def __init__(self,Kp,Ki,Kd):
        
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.previous_error = 0
        self.integral = 0
        
    def compute(self,error, dt):
        # Calculate error
        # Proportional term
        P_out = self.Kp*error
        # Integral term
        self.integral += error*dt
        I_out = self.Ki*self.integral
        # Derivative term 
        derivative = (error-self.previous_error)/dt
        D_out = self.Kd*derivative
        
        # Comptue total output
        output = P_out+I_out+D_out
        
        # Update the previou error
        self.previous_error = error
        return output
    def update_PD(self,error):
        P_out = self.Kp*error
        D_out = self.Kd*(error-self.previous_error)

        self.previous_error = error
        return P_out+D_out
    
    def get_specs(self):
        msg = f'\n KP: {self.Kp}\n KI: {self.Ki}\n KD: {self.Kd}'
        return msg
        

# class PIDController:
#     def __init__(self, Kp=0, Kd=0, Ki=0, set_point=0):
#         self.Kp = Kp
#         self.Kd = Kd
#         self.Ki = Ki
#         self.set_point = set_point

#         self.previous_error = 0
#         self.integral = 0

#     def update_PID(self, error, dt):
#         # Define the error
#         error = error-self.previous_error

#         # Proportional term 
#         P_out = self.Kp*error
#         # Derivative Term 
#         D_out = error/dt *self.Kd
#         # Integral term 
#         self.integral += self.Ki*self.integral
#         I_out = self.Ki*self.integral

#         # Update error 
#         self.previous_error = error
#         # Get total output
#         return P_out+D_out+I_out

#     def setPoint(self,set_point):
#         self.set_point = set_point
#         self.previous_errror = 0
    
#     def update_PD(self,error):
#         P_out = self.Kp*error
#         D_out = self.Kd*(error-self.previous_error)

#         self.previous_error = error
#         return P_out+D_out