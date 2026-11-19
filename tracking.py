import numpy as np



class CustomizedKalmanFilter():
    def __init__(self, dt, is3D, init_uncertainty=1000, x_init_measurement_uncertainty=0.5, y_init_measurement_uncertainty=0.5, z_init_measurement_uncertainty=0.5, x_init_process_variance=0.1, y_init_process_variance=0.1, z_init_process_variance=0.1):
        no_of_states = 6
        if is3D:
            no_of_states = 9
        
        self.x = np.zeros((no_of_states, 1), dtype=float)

        self.P = np.identity(no_of_states) * init_uncertainty
        
        self.u = np.zeros((no_of_states,1))



        self.F = None
        self.H = None

        if is3D:
            self.F = np.array([[1, 0, 0, dt, 0, 0, 0.5*(dt**2), 0, 0],
                               [0, 1, 0, 0, dt, 0, 0, 0.5*(dt**2), 0],
                               [0, 0, 1, 0, 0, dt, 0, 0, 0.5*(dt**2)],
                               [0, 0, 0, 1, 0, 0, dt, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0, dt, 0],
                               [0, 0, 0, 0, 0, 1, 0, 0, dt],
                               [0, 0, 0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 1]])

            self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0, 0, 0]])
            self.R = np.diag([x_init_measurement_uncertainty, y_init_measurement_uncertainty, z_init_measurement_uncertainty])
            A = (dt ** 5) / 20
            B = (dt ** 4) / 8
            C = (dt ** 3) / 6
            D = (dt ** 3) / 3
            E = (dt ** 2) / 2
            self.Q = np.array([
                [x_init_process_variance * A, 0, 0, x_init_process_variance * B, 0, 0, x_init_process_variance * C, 0, 0],
                [0, y_init_process_variance * A, 0, 0, y_init_process_variance * B, 0, 0, y_init_process_variance * C, 0],
                [0, 0, z_init_process_variance * A, 0, 0, z_init_process_variance * B, 0, 0, z_init_process_variance * C],
                [x_init_process_variance * B, 0, 0, x_init_process_variance * D, 0, 0, x_init_process_variance * E, 0, 0],
                [0, y_init_process_variance * B, 0, 0, y_init_process_variance * D, 0, 0, y_init_process_variance * E, 0],
                [0, 0, z_init_process_variance * B, 0, 0, z_init_process_variance * D, 0, 0, z_init_process_variance * E],
                [x_init_process_variance * C, 0, 0, x_init_process_variance * E, 0, 0, x_init_process_variance * dt, 0, 0],
                [0, y_init_process_variance * C, 0, 0, y_init_process_variance * E, 0, 0, y_init_process_variance * dt, 0],
                [0, 0, z_init_process_variance * C, 0, 0, z_init_process_variance * E, 0, 0, z_init_process_variance * dt]
            ])
        else: 
            self.F = np.array([[1, 0, dt, 0, 0.5*(dt**2), 0],
              [0, 1, 0, dt, 0, 0.5*(dt**2)],
              [0, 0, 1, 0, dt, 0],
              [0, 0, 0, 1, 0, dt],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])

            self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]], dtype=float)

            self.R = np.diag([x_init_measurement_uncertainty, y_init_measurement_uncertainty])
            self.Q = np.array([
                [x_init_process_variance * ((dt ** 5) / 20), 0, x_init_process_variance * ((dt ** 4) / 8), 0, x_init_process_variance * ((dt ** 3) / 6), 0],
                [0, y_init_process_variance * ((dt ** 5) / 20), 0, y_init_process_variance * ((dt ** 4) / 8), 0, y_init_process_variance * ((dt ** 3) / 6)],
                [x_init_process_variance * ((dt ** 4) / 8), 0, x_init_process_variance * ((dt ** 3) / 3), 0, x_init_process_variance * ((dt ** 2) / 2), 0],
                [0, y_init_process_variance * ((dt ** 4) / 8), 0, y_init_process_variance * ((dt ** 3) / 3), 0, y_init_process_variance * ((dt ** 2) / 2)],
                [x_init_process_variance * ((dt ** 3) / 6), 0, x_init_process_variance * ((dt ** 2) / 2), 0, x_init_process_variance * dt, 0],
                [0, y_init_process_variance * ((dt ** 3) / 6), 0, y_init_process_variance * ((dt ** 2) / 2), 0, y_init_process_variance * dt]
            ])

        # self.R = np.identity(no_of_states // 3) * init_measurement_uncertainty

        self.I = np.eye(no_of_states)

    def update(self, measurement, x_prime, P_prime):
        y = measurement - self.H @ x_prime
        S = self.H @ P_prime @ np.transpose(self.H) + self.R
        K = P_prime @ np.transpose(self.H) @ np.linalg.pinv(S)
        return y, S, K

    def predict(self):
        x_prime = self.F @ self.x + self.u
        P_prime = self.F @ self.P @ np.transpose(self.F) + self.Q
        return x_prime, P_prime

    def update_cycle(self, new_measurement):
        x_prime, P_prime = self.predict()
        if new_measurement is not None:
            new_measurement = new_measurement.reshape(-1, 1)
            y, S, K = self.update(new_measurement, x_prime, P_prime)
            self.x = x_prime + K @ y
            # print(new_measurement.shape, (self.H @ self.x).shape, y.shape)
            self.P = (self.I - K @ self.H) @ P_prime
        else:
            self.x = x_prime
            self.P = P_prime

    def get_x(self):
        return self.x

    def get_P(self):
        return self.P