from keras.src import ops
from keras.src.backend import standardize_dtype
from keras.src.layers import BatchNormalization


class BatchRenormalization(BatchNormalization):
    # Implemented based on Keras 3.3.2
    def __init__(
        self,
        r_max=3.0,
        d_max=5.0,
        warmup_steps=5000,
        change_d_steps=25000,
        change_r_steps=40000,
        **kwargs,
    ):
        """
        Batch Renormalization (BatchRenormalization) layer.

        Args:
            r_max: Initial maximum allowed value of the correction multiplier r.
            d_max: Initial maximum allowed value of the correction offset d.
            warmup_steps: Steps after which r_max and d_max begin to change.
            change_d_steps: Steps after which d_max reaches its final value.
            change_r_steps: Steps after which r_max reaches its final value.
            **kwargs: Base layer keyword arguments (e.g. `momentum`, `epsilon`).
        """
        super().__init__(**kwargs)
        self.r_max = r_max
        self.d_max = d_max
        self.warmup_steps = warmup_steps
        self.change_d_steps = change_d_steps
        self.change_r_steps = change_r_steps

    def build(self, input_shape):
        self.current_step = self.add_weight(
            (),
            name="current_step",
            initializer="zeros",
            trainable=False,
            autocast=False,
        )
        super().build(input_shape)

    def _compute_r_d_max(self):
        """
        Compute the current values of r_max and d_max based on training steps.
        """

        # Compute r_max change schedule
        r_slope = (self.r_max - 1) / (self.change_r_steps - self.warmup_steps)
        cur_r_max = ops.cond(
            self.current_step < self.warmup_steps,  # Condition
            lambda: 1.0,  # If current_step < warmup_steps, set r_max = 1
            lambda: ops.minimum(
                self.r_max, 1.0 + r_slope * (self.current_step - self.warmup_steps)
            ),  # Else, compute the schedule
        )

        # Compute d_max change schedule
        d_slope = (self.d_max - 0) / (self.change_d_steps - self.warmup_steps)
        cur_d_max = ops.cond(
            self.current_step < self.warmup_steps,  # Condition
            lambda: 0.0,  # If current_step < warmup_steps, set d_max = 0
            lambda: ops.minimum(
                self.d_max, 0.0 + d_slope * (self.current_step - self.warmup_steps)
            ),  # Else, compute the schedule
        )

        return cur_r_max, cur_d_max

    def call(self, inputs, training=None, mask=None):
        if mask is not None and self.synchronized:
            raise ValueError("Cannot pass mask when synchronized")
        input_dtype = standardize_dtype(inputs.dtype)
        if input_dtype in ("float16", "bfloat16"):
            # BN is prone to overflowing for float16/bfloat16 inputs, so we opt
            # out BN for mixed precision.
            inputs = ops.cast(inputs, "float32")

        if training and self.trainable:
            batch_mean, batch_variance = self._moments(inputs, mask)
            moving_mean = ops.cast(self.moving_mean, inputs.dtype)
            moving_variance = ops.cast(self.moving_variance, inputs.dtype)

            self.moving_mean.assign(
                moving_mean * self.momentum + batch_mean * (1.0 - self.momentum)
            )
            self.moving_variance.assign(
                moving_variance * self.momentum + batch_variance * (1.0 - self.momentum)
            )

            r_max, d_max = self._compute_r_d_max()
            r = ops.clip(
                ops.sqrt(batch_variance / (moving_variance + self.epsilon)),
                1 / r_max,
                r_max,
            )
            d = ops.clip(
                (batch_mean - moving_mean) / ops.sqrt(moving_variance + self.epsilon),
                -d_max,
                d_max,
            )
            r = ops.stop_gradient(r)
            d = ops.stop_gradient(d)
            # TODO マルチGPUだとうまく動作しないかも
            self.current_step.assign_add(1.0)
        else:
            batch_mean = ops.cast(self.moving_mean, inputs.dtype)
            batch_variance = ops.cast(self.moving_variance, inputs.dtype)
            r = 1.0
            d = 0.0

        if self.scale:
            gamma = ops.cast(self.gamma, inputs.dtype)
        else:
            gamma = None

        if self.center:
            beta = ops.cast(self.beta, inputs.dtype)
        else:
            beta = None

        outputs = ops.batch_normalization(
            x=inputs,
            mean=batch_mean,
            variance=batch_variance,
            axis=self.axis,
            offset=d,
            scale=r,
            epsilon=self.epsilon,
        )
        outputs = outputs * gamma + beta
        return ops.cast(outputs, self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "r_max": self.r_max,
                "d_max": self.d_max,
                "warmup_steps": self.warmup_steps,
                "change_d_steps": self.change_d_steps,
                "change_r_steps": self.change_r_steps,
            }
        )
        return config
