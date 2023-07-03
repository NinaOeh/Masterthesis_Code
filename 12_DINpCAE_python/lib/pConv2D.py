import tensorflow as tf

class pConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size = (3,3), 
                 strides = (1,1), use_bias = True, activation = None,
                 regularizer = None, **kwargs):
      super(pConv2D, self).__init__() 
      
      """
      Including partial convolutions to the original 2D convolutional layer.
      
      Parameters
      ----------
      filters : int
          Number of filters in the convolution.
      kernel_size : int or tuple/list of 2 int, optional
          Size of the convolution kernel. The default is (3,3).
      strides : int or tuple/list of 2 int, optional
          Strides of the convolution. The default is (1,1).
      use_bias : bool, optional
          Whether the layer uses a bias vector. The default is True.
      activation : str, optional
          Name of the activation function to use. The default is None.
      regularizer : str, optional
          Name of the regularizer to use. The default is None.

      """
      
      self.filters = filters
      self.kernel_size = kernel_size
      self.strides = strides
      self.use_bias = use_bias
      
      self.in_activation = activation
      if activation is None:
        self.activation = None
      elif isinstance(activation, str):
        self.activation = tf.keras.activations.deserialize(activation)
      elif callable(activation):
        self.activation = activation
      elif isinstance(activation, dict):
        self.activation = tf.keras.layers.deserialize(activation)
      else:
        raise ValueError("Failed to interpret activation arg, expected None, string or instance of tf.keras.layers.Layer.")
      
      self.regularization = regularizer
        
    def get_config(self):
     
      if isinstance(self.activation, tf.keras.layers.Layer):
        act = self.activation.get_config()
      elif callable(self.activation):
        act = tf.keras.activations.serialize(self.activation)
        
      return {"filters": self.filters, 
              "kernel_size": self.kernel_size,
              "strides": self.strides,
              "use_bias": self.use_bias,
              "activation" : act,
              "regularizer" : self.regularization,
              } # TODO: test

    def build(self, input_shape):       
      channel_axis = -1 # assume channel_last data format
      if len(input_shape) != 2:
        raise ValueError("Invalid input shape, expected length-two list containing image and mask.")
      if input_shape[0][channel_axis] is None:
        raise ValueError("The channel dimension of image inputs should be defined. Found `None`")
      if input_shape[1][channel_axis] is None:
        raise ValueError("The channel dimension of mask inputs should be defined. Found `None`.")

      self.input_dim_img = input_shape[0][channel_axis]
      print(self.input_dim_img)
      
      self.input_dim_mask = input_shape[1][channel_axis]
      print(self.input_dim_mask)
      
      # Image kernel
      print(self.kernel_size)
      print((self.input_dim_img, self.filters))
      kernel_shape_img = tuple(self.kernel_size)  + (self.input_dim_img, self.filters) #tuple
      print(f"Kernel SHAPE IMG: {kernel_shape_img}")
      # add regularization
      self.kernel = self.add_weight(shape=kernel_shape_img,
                                    initializer=tf.keras.initializers.GlorotUniform(),
                                    #regularizer=self.regularization,
                                    name='pConv2D_kernel')
      # Mask kernel
      kernel_shape_mask = tuple(self.kernel_size)  + (self.input_dim_mask, self.filters) #tuple
      print(f"Kernel SHAPE MASK: {kernel_shape_mask}")
      self.kernel_mask = tf.ones(kernel_shape_mask)


      # Calculate padding sizes per dimension
      p0 = int((self.kernel_size[0]-1)/2)
      p1 = int((self.kernel_size[1]-1)/2)
      self.paddings = tf.constant([[0, 0], 
                              [p0, p0], 
                              [p1, p1], 
                              [0, 0]])
        
      
      if self.use_bias:
        # add regularization
        self.bias = self.add_weight(shape=(self.filters,), 
                                    initializer=tf.keras.initializers.zeros(), 
                                    #regularizer=self.regularization,
                                    name='bias')
      else:
        self.bias = None        
      self.built = True

    def call(self, inputs):
      # Both image and mask must be supplied
      if type(inputs) is not list or len(inputs) != 2:
        raise Exception('pConv2D must be called on a list of two tensors [img, mask]. Instead got: ' + str(inputs))

      # Padding (=zero padding) done explicitly so that padding becomes part of the masked partial convolution
      imgs  = tf.pad(inputs[0], self.paddings, "CONSTANT", constant_values=0) # zero padding
      masks = tf.pad(inputs[1], self.paddings, "CONSTANT", constant_values=0) # zero padding
      
      mask_output = tf.nn.conv2d(masks, self.kernel_mask, strides = (1,) + tuple(self.strides) + (1,), 
                                 padding = "VALID") 
      img_output = tf.nn.conv2d(imgs, self.kernel, strides = (1,) + tuple(self.strides) + (1,), 
                                padding = "VALID") 
     
      # Calculate the mask ratio on each pixel in the output mask
      n = self.kernel_size[0] * self.kernel_size[1] 
      mask_ratio = n / (mask_output + 1e-8) # avoid devision by zero

      # Clip output to be between 0 and 1
      mask_output = tf.clip_by_value(mask_output, 0, 1)

      # Remove ratio values where there are holes
      mask_ratio = mask_ratio * mask_output

      # Normalize image output
      img_output = img_output * mask_ratio

      # Apply bias only to the image (if chosen to do so)
      if self.use_bias:
        img_output = tf.nn.bias_add(img_output, self.bias) # data format?
        
      # Call activation function
      if not self.activation is None:
        img_output = self.activation(img_output)
          
      return [img_output, mask_output]
    
