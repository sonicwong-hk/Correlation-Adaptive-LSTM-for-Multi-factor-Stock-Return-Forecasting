import math
import random
import numpy as np

# ---------- Helper functions ----------

def zero_vector(n):
    return [0.0 for _ in range(n)]

def zero_matrix(rows, cols):
    return np.zeros((rows, cols))

def vector_dot(a, b):
    if len(a) != len(b):
        raise ValueError("Dot product: length mismatch")
    return sum(x * y for x, y in zip(a, b))

def vector_add(a, b):
    return [x + y for x, y in zip(a, b)]

def subtract_vector(a, b):
    return [x - y for x, y in zip(a, b)]

def multiply_vector(vec, scalar):
    return [scalar * x for x in vec]

def add_vector(a, b):
    return [x + y for x, y in zip(a, b)]

def elementwise_mul(a, b):
    return [x * y for x, y in zip(a, b)]

def mat_vec_mul(matrix, vec):
    return [vector_dot(row, vec) for row in matrix]

def outer_product(vec1, vec2):
    return np.outer(vec1, vec2)

def add_matrix(A, B):
    return A + B

def subtract_matrix(A, B):
    return A - B

def multiply_matrix(matrix, scalar):
    return matrix * scalar

# ---------- Activation Functions ----------

def sigmoid(x):
    x = max(-700, min(700, x))  # Clamp value to avoid overflow
    return 1 / (1 + math.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def tanh(x):
    return math.tanh(x)

def dtanh(y):
    return 1 - y * y

# ---------- Compute Gate ----------

def compute_gate(W, U, b, x, h, activation):
    z_x = mat_vec_mul(W, x)      # vector of length hidden_size
    z_h = mat_vec_mul(U, h)      # vector of length hidden_size
    z_temp = vector_add(z_x, z_h)
    if len(z_temp) != len(b):
        raise ValueError(f"Dimension mismatch in compute_gate: {len(z_temp)} vs {len(b)}")
    z = vector_add(z_temp, b)
    return [activation(val) for val in z]

def xavier_init(shape):
    fan_in = shape[1]
    fan_out = shape[0]
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape)

# ---------- LSTMCell ----------

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Xavier initialization for weight matrices and zeros for biases
        self.W_i = xavier_init((hidden_size, input_size))
        self.U_i = xavier_init((hidden_size, hidden_size))
        self.b_i = np.zeros(hidden_size)

        self.W_f = xavier_init((hidden_size, input_size))
        self.U_f = xavier_init((hidden_size, hidden_size))
        self.b_f = np.zeros(hidden_size)

        self.W_o = xavier_init((hidden_size, input_size))
        self.U_o = xavier_init((hidden_size, hidden_size))
        self.b_o = np.zeros(hidden_size)

        self.W_g = xavier_init((hidden_size, input_size))
        self.U_g = xavier_init((hidden_size, hidden_size))
        self.b_g = np.zeros(hidden_size)

        # Initialize Adam optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

        # Initialize moment estimates
        self.m_W_i = np.zeros_like(self.W_i)
        self.v_W_i = np.zeros_like(self.W_i)
        self.m_U_i = np.zeros_like(self.U_i)
        self.v_U_i = np.zeros_like(self.U_i)
        self.m_b_i = np.zeros_like(self.b_i)
        self.v_b_i = np.zeros_like(self.b_i)

        self.m_W_f = np.zeros_like(self.W_f)
        self.v_W_f = np.zeros_like(self.W_f)
        self.m_U_f = np.zeros_like(self.U_f)
        self.v_U_f = np.zeros_like(self.U_f)
        self.m_b_f = np.zeros_like(self.b_f)
        self.v_b_f = np.zeros_like(self.b_f)

        self.m_W_o = np.zeros_like(self.W_o)
        self.v_W_o = np.zeros_like(self.W_o)
        self.m_U_o = np.zeros_like(self.U_o)
        self.v_U_o = np.zeros_like(self.U_o)
        self.m_b_o = np.zeros_like(self.b_o)
        self.v_b_o = np.zeros_like(self.b_o)

        self.m_W_g = np.zeros_like(self.W_g)
        self.v_W_g = np.zeros_like(self.W_g)
        self.m_U_g = np.zeros_like(self.U_g)
        self.v_U_g = np.zeros_like(self.U_g)
        self.m_b_g = np.zeros_like(self.b_g)
        self.v_b_g = np.zeros_like(self.b_g)

        self.timestep = 0
        self.reset_gradients()

    def reset_gradients(self):
        self.dW_i = np.zeros((self.hidden_size, self.input_size))
        self.dU_i = np.zeros((self.hidden_size, self.hidden_size))
        self.db_i = np.zeros(self.hidden_size)

        self.dW_f = np.zeros((self.hidden_size, self.input_size))
        self.dU_f = np.zeros((self.hidden_size, self.hidden_size))
        self.db_f = np.zeros(self.hidden_size)

        self.dW_o = np.zeros((self.hidden_size, self.input_size))
        self.dU_o = np.zeros((self.hidden_size, self.hidden_size))
        self.db_o = np.zeros(self.hidden_size)

        self.dW_g = np.zeros((self.hidden_size, self.input_size))
        self.dU_g = np.zeros((self.hidden_size, self.hidden_size))
        self.db_g = np.zeros(self.hidden_size)

    def forward(self, x, h_prev, c_prev, weight=1.0):
        if random.random() < 0.2:
            x = [0.0] * len(x)
        i = compute_gate(self.W_i, self.U_i, self.b_i, x, h_prev, sigmoid)
        f = compute_gate(self.W_f, self.U_f, self.b_f, x, h_prev, sigmoid)
        o = compute_gate(self.W_o, self.U_o, self.b_o, x, h_prev, sigmoid)
        g = compute_gate(self.W_g, self.U_g, self.b_g, x, h_prev, tanh)

        c = vector_add(elementwise_mul(f, c_prev), elementwise_mul(i, g))
        c = [weight * ci for ci in c]
        h = elementwise_mul(o, [tanh(ci) for ci in c])

        cache = {
            "x": x,
            "h_prev": h_prev,
            "c_prev": c_prev,
            "i": i,
            "f": f,
            "o": o,
            "g": g,
            "c": c,
            "h": h
        }
        return h, c, cache

    def backward(self, d_h, d_c, cache):
        x = cache["x"]
        h_prev = cache["h_prev"]
        c_prev = cache["c_prev"]
        i = cache["i"]
        f = cache["f"]
        o = cache["o"]
        g = cache["g"]
        c = cache["c"]

        tanh_c = [tanh(ci) for ci in c]
        d_o = [d_h_i * tanh_c_i for d_h_i, tanh_c_i in zip(d_h, tanh_c)]
        d_c_local = [d_h_i * o_i * (1 - tanh_c_i**2) for d_h_i, o_i, tanh_c_i in zip(d_h, o, tanh_c)]
        d_c_total = [d_local + dc for d_local, dc in zip(d_c_local, d_c)]

        d_i = elementwise_mul(d_c_total, g)
        d_g = elementwise_mul(d_c_total, i)
        d_f = elementwise_mul(d_c_total, c_prev)
        d_c_prev = elementwise_mul(d_c_total, f)

        d_z_i = [di * dsigmoid(ii) for di, ii in zip(d_i, i)]
        d_z_f = [df * dsigmoid(fi) for df, fi in zip(d_f, f)]
        d_z_o = [do * dsigmoid(oi) for do, oi in zip(d_o, o)]
        d_z_g = [dg * dtanh(gi) for dg, gi in zip(d_g, g)]

        # Ensure bias gradients maintain correct shape
        assert self.db_i.shape == (self.hidden_size,), f"db_i shape mismatch: {self.db_i.shape}"
        assert self.db_f.shape == (self.hidden_size,), f"db_f shape mismatch: {self.db_f.shape}"
        assert self.db_o.shape == (self.hidden_size,), f"db_o shape mismatch: {self.db_o.shape}"
        assert self.db_g.shape == (self.hidden_size,), f"db_g shape mismatch: {self.db_g.shape}"

        dW_i_inc = outer_product(d_z_i, x)
        dU_i_inc = outer_product(d_z_i, h_prev)
        self.dW_i += dW_i_inc
        self.dU_i += dU_i_inc
        self.db_i += np.array(d_z_i, dtype=np.float64).flatten()

        dW_f_inc = outer_product(d_z_f, x)
        dU_f_inc = outer_product(d_z_f, h_prev)
        self.dW_f += dW_f_inc
        self.dU_f += dU_f_inc
        self.db_f += np.array(d_z_f, dtype=np.float64).flatten()

        dW_o_inc = outer_product(d_z_o, x)
        dU_o_inc = outer_product(d_z_o, h_prev)
        self.dW_o += dW_o_inc
        self.dU_o += dU_o_inc
        self.db_o += np.array(d_z_o, dtype=np.float64).flatten()

        dW_g_inc = outer_product(d_z_g, x)
        dU_g_inc = outer_product(d_z_g, h_prev)
        self.dW_g += dW_g_inc
        self.dU_g += dU_g_inc
        self.db_g += np.array(d_z_g, dtype=np.float64).flatten()

        dx_i = mat_trans_vec_mul(self.W_i, d_z_i)
        dx_f = mat_trans_vec_mul(self.W_f, d_z_f)
        dx_o = mat_trans_vec_mul(self.W_o, d_z_o)
        dx_g = mat_trans_vec_mul(self.W_g, d_z_g)
        dx = [a + b + c + d for a, b, c, d in zip(dx_i, dx_f, dx_o, dx_g)]

        dh_i = mat_trans_vec_mul(self.U_i, d_z_i)
        dh_f = mat_trans_vec_mul(self.U_f, d_z_f)
        dh_o = mat_trans_vec_mul(self.U_o, d_z_o)
        dh_g = mat_trans_vec_mul(self.U_g, d_z_g)
        dh_prev = [a + b + c + d for a, b, c, d in zip(dh_i, dh_f, dh_o, dh_g)]

        return dx, dh_prev, d_c_prev

    def update_parameters(self, learning_rate):
        self.timestep += 1

        def adam_update(param, grad, m, v):
            grad = np.array(grad, dtype=np.float64)
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            m_hat = m / (1 - self.beta1 ** self.timestep)
            v_hat = v / (1 - self.beta2 ** self.timestep)
            param -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            return param, m, v

        self.W_i, self.m_W_i, self.v_W_i = adam_update(self.W_i, self.dW_i, self.m_W_i, self.v_W_i)
        self.U_i, self.m_U_i, self.v_U_i = adam_update(self.U_i, self.dU_i, self.m_U_i, self.v_U_i)
        self.b_i, self.m_b_i, self.v_b_i = adam_update(self.b_i, self.db_i, self.m_b_i, self.v_b_i)

        self.W_f, self.m_W_f, self.v_W_f = adam_update(self.W_f, self.dW_f, self.m_W_f, self.v_W_f)
        self.U_f, self.m_U_f, self.v_U_f = adam_update(self.U_f, self.dU_f, self.m_U_f, self.v_U_f)
        self.b_f, self.m_b_f, self.v_b_f = adam_update(self.b_f, self.db_f, self.m_b_f, self.v_b_f)

        self.W_o, self.m_W_o, self.v_W_o = adam_update(self.W_o, self.dW_o, self.m_W_o, self.v_W_o)
        self.U_o, self.m_U_o, self.v_U_o = adam_update(self.U_o, self.dU_o, self.m_U_o, self.v_U_o)
        self.b_o, self.m_b_o, self.v_b_o = adam_update(self.b_o, self.db_o, self.m_b_o, self.v_b_o)

        self.W_g, self.m_W_g, self.v_W_g = adam_update(self.W_g, self.dW_g, self.m_W_g, self.v_W_g)
        self.U_g, self.m_U_g, self.v_U_g = adam_update(self.U_g, self.dU_g, self.m_U_g, self.v_U_g)
        self.b_g, self.m_b_g, self.v_b_g = adam_update(self.b_g, self.db_g, self.m_b_g, self.v_b_g)

        self.reset_gradients()

# Additional helper: Multiply the transpose of a matrix with a vector.
def mat_trans_vec_mul(matrix, vec):
    cols = len(matrix[0])
    result = [0.0] * cols
    for i in range(len(matrix)):
        for j in range(cols):
            result[j] += matrix[i][j] * vec[i]
    return result

# ---------- LSTM with Output Layer for Regression ----------

class LSTM:
    def __init__(self, input_size, hidden_size, output_size=1):
        self.cell = LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W_hidden = np.random.uniform(-1, 1, (hidden_size, hidden_size//2))
        self.b_hidden = np.random.uniform(-1, 1, hidden_size//2)
        self.W_out = np.random.uniform(-1, 1, (hidden_size//2, output_size))
        self.b_out = np.random.uniform(-1, 1, output_size)
        self.dW_hidden = np.zeros_like(self.W_hidden)
        self.db_hidden = np.zeros_like(self.b_hidden)
        self.dW_out = np.zeros_like(self.W_out)
        self.db_out = np.zeros_like(self.b_out)
        self.h = [0.0] * hidden_size
        self.c = [0.0] * hidden_size

    def reset_state(self):
        self.h = [0.0] * self.hidden_size
        self.c = [0.0] * self.hidden_size

    def forward(self, inputs, weight):
        outputs = []
        caches = []
        for x in inputs:
            self.h, self.c, cache = self.cell.forward(x, self.h, self.c, weight)
            outputs.append(self.h)
            caches.append(cache)
        h_final = outputs[-1]
        hidden_out = [max(0, sum(w * h for w, h in zip(row, h_final)) + b) for row, b in zip(self.W_hidden, self.b_hidden)]
        y_pred = sum(w * h for w, h in zip(self.W_out.T[0], hidden_out)) + self.b_out
        return y_pred, outputs, caches

    def backward(self, d_y, outputs, caches):
        h_final = outputs[-1]
        hidden_out = [max(0, sum(w * h for w, h in zip(row, h_final)) + b) for row, b in zip(self.W_hidden, self.b_hidden)]

        d_hidden_out = [d_y * w for w in self.W_out.T[0]]
        self.db_out += d_y
        for j in range(len(self.W_out)):
            self.dW_out[j, 0] += d_y * hidden_out[j]

        d_h_final = [0.0] * self.hidden_size
        for j in range(len(hidden_out)):
            if hidden_out[j] > 0:
                self.db_hidden[j] += d_hidden_out[j]
                for i in range(self.hidden_size):
                    d_h_final[i] += d_hidden_out[j] * self.W_hidden[i][j]
                    self.dW_hidden[i][j] += d_hidden_out[j] * h_final[i]

        d_outputs = [[0.0] * self.hidden_size for _ in caches]
        d_outputs[-1] = d_h_final
        dh_next = [0.0] * self.hidden_size
        dc_next = [0.0] * self.hidden_size
        for t in reversed(range(len(caches))):
            d_h = [d_o + d_n for d_o, d_n in zip(d_outputs[t], dh_next)]
            _, dh_next, dc_next = self.cell.backward(d_h, dc_next, caches[t])

    def update_parameters(self, learning_rate):
        self.cell.update_parameters(learning_rate)
        self.W_hidden -= learning_rate * self.dW_hidden
        self.b_hidden -= learning_rate * self.db_hidden
        self.W_out -= learning_rate * self.dW_out
        self.b_out -= learning_rate * self.db_out
        self.dW_hidden = np.zeros_like(self.W_hidden)
        self.db_hidden = np.zeros_like(self.b_hidden)
        self.dW_out = np.zeros_like(self.W_out)
        self.db_out = np.zeros_like(self.b_out)

# ---------- Softmax and Cross-Entropy Loss Functions ----------

def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def cross_entropy_loss(logits, target_index):
    probs = softmax(logits)
    loss = -math.log(probs[target_index] + 1e-12)
    return loss, probs

def cross_entropy_derivative(probs, target_index):
    one_hot = [0.0 for _ in probs]
    one_hot[target_index] = 1.0
    return [p - o for p, o in zip(probs, one_hot)]