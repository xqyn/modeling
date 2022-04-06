# Modeling
Author: QX

token = ''
username = 'xqyn'
useremail = 'nxquy.bms@gmail.com'
repo = 'Modeling/'
!git clone https://{token}@github.com/{username}/{repo}


!git config --global user.name {username}

!git config --global user.email {useremail}

%cd {repo}

!git remote -v

!git add .

!git push origin 

