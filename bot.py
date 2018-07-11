import cv2
from darkflow.net.build import TFNet
import imageio
from io import BytesIO
import logging
import os
from telegram.ext import Updater, MessageHandler, Filters

options = {
    "model": "cfg/yolov2-tiny-voc.cfg",
    "load": "bin/yolov2-tiny-voc.weights",
    "threshold": 0.1
}
tfnet = TFNet(options)


def tuple_coord(dict_coord):
    return (dict_coord['x'], dict_coord['y'])


def detect(bot, update):
    logging.info('received message')
    image = imageio.imread(update.message.photo)
    detected_objects = tfnet.return_predict(image)
    for detected in detected_objects:
        cv2.rectangle(image,
                      tuple_coord(detected['topleft']),
                      tuple_coord(detected['bottomright']),
                      (0, 255, 0))

    pseudo_file = BytesIO()
    pseudo_file.name = 'image.jpg'
    imageio.imwrite(pseudo_file, image)
    pseudo_file.seek(0)

    update.message.reply_photo(photo=pseudo_file)


updater = Updater(os.environ['TELEGRAM_TOKEN'])

updater.dispatcher.add_handler(MessageHandler(Filters.photo, detect))

updater.start_polling()
updater.idle()
