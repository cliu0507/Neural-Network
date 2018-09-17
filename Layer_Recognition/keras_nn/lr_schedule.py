#Learning Rate scheduler function
def lr_scheduler_50(epoch):
	"""Learning Rate Schedule
	Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
	Called automatically every epoch as part of callbacks during training.
	# Arguments
		epoch (int): The number of epochs
	# Returns
		lr (float32): learning rate
	"""
	lr = 1e-3
	if epoch > 40:
		lr = 1e-5
	elif epoch > 25:
		lr = 1e-4
	print('Learning rate: ', lr)
	return lr


def lr_scheduler_100(epoch):
	"""Learning Rate Schedule
	Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
	Called automatically every epoch as part of callbacks during training.
	# Arguments
		epoch (int): The number of epochs
	# Returns
		lr (float32): learning rate
	"""
	lr = 1e-3
	if epoch > 80:
		lr = 1e-5
	elif epoch > 50:
		lr = 1e-4
	print('Learning rate: ', lr)
	return lr


def lr_scheduler_200(epoch):
	"""Learning Rate Schedule
	Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
	Called automatically every epoch as part of callbacks during training.
	# Arguments
		epoch (int): The number of epochs
	# Returns
		lr (float32): learning rate
	"""
	lr = 1e-3
	if epoch > 150:
		lr = 1e-6
	elif epoch > 120:
		lr = 1e-5
	elif epoch > 80:
		lr = 1e-4
	print('Learning rate: ', lr)
	return lr