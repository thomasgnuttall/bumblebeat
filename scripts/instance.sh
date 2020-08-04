# Commands for interacting with the instance

# SSH
ssh -i ~/.ssh/aws_key.pem ec2-user@ec2-18-220-186-142.us-east-2.compute.amazonaws.com

# Realtime sync of local directory to instance
fswatch bumblebeat/ | xargs -I {} rsync -avz -e "ssh -i ~/.ssh/aws_key.pem" bumblebeat/ ec2-user@ec2-18-220-186-142.us-east-2.compute.amazonaws.com:bumblebeat

# Copy folder from instance to local (-r for folder)
scp -r -i ~/.ssh/aws_key.pem ec2-user@ec2-18-220-186-142.us-east-2.compute.amazonaws.com:gpu_run-groove/full-midionly/20200630-220722/log.txt .

# Flush CUDA memory (might need root access)
rmmod nvidia 
modprobe nvidia