# Commands for interacting with the instance

# sync local directory to instance
fswatch bumblebeat/ | xargs -I {} rsync -avz -e "ssh -i ~/.ssh/aws_key.pem" bumblebeat/ ec2-user@ec2-3-23-115-196.us-east-2.compute.amazonaws.com:bumblebeat

# Copy folder from instance to local (-r is for folder)
scp -r -i ~/.ssh/aws_key.pem ec2-user@ec2-3-15-183-86.us-east-2.compute.amazonaws.com:bumblebeat/gpu_run-groove/ .

# Flush CUDA memory (might need root access)
rmmod nvidia 
modprobe nvidia