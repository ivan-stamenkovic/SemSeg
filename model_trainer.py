import tensorflow as tf
import os.path

from global_settings import GLOBAL
import image_preprocessor
import model_runner


correct_label = tf.placeholder(tf.float32, [None, GLOBAL.IMAGE_SHAPE[0], GLOBAL.IMAGE_SHAPE[1], GLOBAL.NUMBER_OF_CLASSES])
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

#Loads vgg and gets tensors from the graph
def load_pretrained_vgg(session):
    model = tf.saved_model.loader.load(session, ['vgg16'], GLOBAL.VGG_PATH)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    layer3 = graph.get_tensor_by_name('layer3_out:0')
    layer4 = graph.get_tensor_by_name('layer4_out:0')
    layer7 = graph.get_tensor_by_name('layer7_out:0')

    return image_input, keep_prob, layer3, layer4, layer7


def add_fcn_layers(layer3, layer4, layer7):
    #1x1 convolution instead of fully connected
    fcn8 = tf.layers.conv2d(layer7, filters=GLOBAL.NUMBER_OF_CLASSES, kernel_size=1, name="fcn8")

    #Upsample to match size of layer4
    fcn9 = tf.layers.conv2d_transpose(fcn8, filters=layer4.get_shape().as_list()[-1],
                                      kernel_size=4, strides=(2, 2), padding='SAME', name="fcn9")

    #Add skip connection
    fcn9_skip_connected = tf.add(fcn9, layer4, name="fcn9_plus_vgg_layer4")

    #Upsample to match size of layer3
    fcn10 = tf.layers.conv2d_transpose(fcn9_skip_connected, filters=layer3.get_shape().as_list()[-1],
                                       kernel_size=4, strides=(2, 2), padding='SAME', name="fcn10_conv2d")

    #Add skip connection
    fcn10_skip_connected = tf.add(fcn10, layer3, name="fcn10_plus_vgg_layer3")

    #Final upsampling
    fcn11 = tf.layers.conv2d_transpose(fcn10_skip_connected, filters=GLOBAL.NUMBER_OF_CLASSES,
                                       kernel_size=16, strides=(8, 8), padding='SAME', name="fcn11")
    return fcn11


def create_optimizer(last_layer, correct_label, learning_rate):
    # Reshape 4D to 2D tensors, row = pixel, column = class
    logits = tf.reshape(last_layer, (-1, GLOBAL.NUMBER_OF_CLASSES), name="fcn_logits")
    correct_label_reshaped = tf.reshape(correct_label, (-1, GLOBAL.NUMBER_OF_CLASSES))

    #Calculate cross entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label_reshaped[:])
    #Reduce mean to get the loss
    loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")
    #Use Adam as optimizer
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")

    return logits, train_op, loss_op


def training(session, batch_function, train_op, loss_op, input_image, correct_label, keep_prob, leearning_rate):
    for epoch in range(GLOBAL.EPOCHS):
        total_loss = 0
        for image_batch, gt_batch in batch_function():
            loss, _ = session.run([loss_op, train_op],
                                feed_dict={input_image: image_batch, correct_label: gt_batch,
                                        keep_prob: GLOBAL.KEEP_PROB_VALUE, learning_rate: GLOBAL.LEARNING_RATE})
            total_loss += loss
        
        print("EPOCH {} ...".format(epoch + 1))
        print("Loss = {:.3f}".format(total_loss))
        print()


def run():
    batch_function = image_preprocessor.create_batch_function()

    with tf.Session() as session:
        print("Building NN Model")
        image_input, keep_prob, layer3, layer4, layer7 = load_pretrained_vgg(session)

        model_output = add_fcn_layers(layer3, layer4, layer7)

        logits, train_op, loss_op = create_optimizer(model_output, correct_label, learning_rate)

        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        print("Model built, initiating training")
        training(session, batch_function, train_op, loss_op, image_input, correct_label, keep_prob, learning_rate)

        print("Training done, generating test images")
        model_runner.run_image_dir(GLOBAL.TEST_IMAGE_DIRECTORY, GLOBAL.TEST_IMAGE_DIRECTORY, session, image_input, logits, keep_prob)
        
        print("Test images generated, saving model for future use")
        saver = tf.train.Saver()
        saver.save(session, os.path.join(GLOBAL.SAVED_MODEL_PATH, GLOBAL.SAVED_MODEL_NAME))
        
        print("Done!")

if __name__ == "__main__":
    run()