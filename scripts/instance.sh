# Commands for interacting with the instance

# SSH
ssh -i ~/.ssh/aws_key.pem ec2-user@ec2-3-23-115-196.us-east-2.compute.amazonaws.com

# Realtime sync of local directory to instance
fswatch bumblebeat/ | xargs -I {} rsync -avz -e "ssh -i ~/.ssh/aws_key.pem" bumblebeat/ ec2-user@ec2-3-23-115-196.us-east-2.compute.amazonaws.com:bumblebeat

# Copy folder from instance to local (remove -r for file)
scp -r -i ~/.ssh/aws_key.pem ec2-user@ec2-3-23-115-196.us-east-2.compute.amazonaws.com:gpu_run-groove/ .

# Flush CUDA memory (might need root access)
rmmod nvidia 
modprobe nvidia