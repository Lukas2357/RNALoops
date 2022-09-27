syncinit() {

    echo -e "\n### Combining GitHub and GDrive sync ###\n\n-> Let's get started...\n"
    echo -e "-> We first try to clone or connect a repo from GitHub"
    echo -e "-> If you have a local folder with the given name, we try to connect, else we clone."
    read -p "-> Do you have access to a repo with given name and like to connect/clone it? (y/n) " ans

    if [[ $ans == [yY] || $ans == [yY][eE][sS] ]]; then

        read -p "-> Please give us the username of that repo: " user

        if [ -d "$1" ]; then

            cd "$1" || return

            if [ -d ".git" ]; then
                echo -e "\n-> Local folder $1 is a git repo."
                echo -e "\n-> Setting $user/$1 as remote on branch main and pull content..."
                git checkout -b main
                git remote add origin git@github.com:$user/$1.git &> /dev/null
                git pull origin main --allow-unrelated-histories
                echo -e "\n-> If it failed, you might abort and retry or add manually. We continue in any case..."
            else
                echo -e "\n-> Local folder $1 is not a git repo. Setting it up with $user/$1"
                git init

                echo -e "\n-> You can now add and commit all files or do this manually in another terminal."
                read -p "-> Do you want to add and commit all files? (y/n): " ans
                if [[ $ans == [yY] || $ans == [yY][eE][sS] ]]; then
                    git add .
                    git commit -m 'initial commit'
                fi

                git remote add origin git@github.com:$user/$1.git &> /dev/null
                git checkout -b main
                git pull origin main --allow-unrelated-histories

                echo -e "\n-> If it failed, you might abort and retry or add manually. We continue in any case..."
            fi

            cd .. || return

        else

            echo -e "\n-> Trying to clone public repo...\n"
            git clone https://github.com/$user/$1.git &> /dev/null

            if [ -d "$1" ]; then
                echo -e "-> Successfull!\n"
            else
                echo -e "\n-> Trying to clone private repo...\n"
                git clone https://$user@github.com/$user/$1 &> /dev/null

                if [ -d "$1" ]; then
                    echo -e "-> Successfull!\n"
                else
                    echo -e "-> Could not clone, are you sure you gave the correct user and have access to the repo?\n"
                    echo "-> Continue without cloning in 10 sec, abort manually if you like to retry."
                    sleep 10
                    if [ ! -d "$1" ]; then
                        mkdir $1
                    fi
                fi
            fi
        fi
    else
        if [ ! -d "$1" ]; then
            mkdir "$1"
        fi
    fi

    cd "$1" || return

    if [ -d ".sync" ]; then
        echo -e "-> Found .sync folder in $1, skip cloning it...\n"
    else
        echo -e "-> Cloning .sync from Lukas2357/GDriveBackup...\n"
        git clone https://github.com/Lukas2357/GDriveBackup.git .sync &> /dev/null
    fi

    cd ".sync" || return
    if [ -d ".git" ]; then
        rm -rf ".git"
    fi
    cd .. || return



    echo -e "-> Sync requires pydrive. If you continue it is installed via pip (if not already installed)."
    echo -e "-> If you prefer manual install or using virtual environment, quit and prepare it, then rerun.\n"
    read -p "-> Continue? (y/n): " confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || return;
    pip install pydrive &> /dev/null
    echo -e "-> Installed pydrive using pip (if not already)\n\n"

    echo "-> Everything prepared. You can now call..."
    echo -e "-> 'bash .sync/sync.sh sync_all' to sync with GitHub + GDrive"
    echo -e "-> 'bash .sync/sync.sh sync_all' to sync with GitHub + create copy (with date) of local in GDrive"
    echo -e "-> 'bash .sync/sync.sh sync' to sync GDrive bidirectional (local->GDrive, GDrive->local)"
    echo -e "-> 'bash .sync/sync.sh sync_down' to just download from GDrive"
    echo -e "-> 'bash .sync/sync.sh sync_up' to just upload to GDrive"
    echo -e "-> 'bash .sync/sync.sh sync_clean' to delete GDrive and replace with local (only if messed up!)\n"

    echo -e "-> For convenience put content of .sync/sync.sh in .bash_functions and call 'sync' etc. directly."

    echo -e "-> Full sync with the cloud will always keep the newer file versions.\n"
    echo -e "-> It is recommended to sync GDrive content now, to be up to date.\n"
    read -p "-> Sync GDrive content now? (y/n) " ans

    if [[ $ans == [yY] || $ans == [yY][eE][sS] ]]; then
        sync "initial sync"
    fi

    echo -e "\n-> You are good to go. Congrats!\n"

}

sync_github() {
    if [ -d ".git" ]; then
        git add .
        git commit -m "$1"
        if [ "$1" == "initial sync" ]; then
          git push --set-upstream origin main
        else
          git push origin main
        fi
    fi
    echo -e "\nEverything up to date :)"
}

prepare_sync() {
    if [ -d ".sync" ]; then
      if [ -d ".git" ]; then
        if [ "$1" == "initial sync" ]; then
          git branch --set-upstream-to=origin/main main
        fi
        git pull origin main
      fi
      cd .sync || return
    else
      echo "You need to be in an initialized folder to sync."
      return
    fi
}

sync_down() {
    prepare_sync "$1"
    python3 sync.py -s .. -dl True
    cd ..
    sync_github "$1"
}

sync_up() {
    prepare_sync "$1"
    python3 sync.py -s .. -ul True
    cd ..
    sync_github "$1"
}

sync_copy() {
    prepare_sync "$1"
    python3 sync.py -s .. -ul True -cp True
    cd ..
    sync_github "$1"
}

sync_clean() {
    prepare_sync "$1"
    python3 sync.py -s .. -c True
    cd ..
    sync_github "$1"
}

sync() {
    prepare_sync "$1"
    python3 sync.py -s ..
    cd ..
    sync_github "$1"
}

"$@"