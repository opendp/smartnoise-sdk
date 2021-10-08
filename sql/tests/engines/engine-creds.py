
'''
 Console utility to set passwords for database connections used by unit tests.
 Iterates through all engines listed in the config file, and prompts for
 passwords.  Passing in --list will list all passwords that are currently set.
 Passwords are stored in the local keyring.
'''

def main():
    import yaml
    from os import path
    import keyring
    import getpass
    import readchar
    import sys
    list_mode = False
    if "--list" in sys.argv:
        list_mode = True
    conns = None
    home = path.expanduser("~")
    p = path.join(home, ".smartnoise", "connections-unit.yaml")
    if not path.exists(p):
        print ("No config file at ~/.smartnoise/connections-unit.yaml")
        return
    with open(p, 'r') as stream:
        conns = yaml.safe_load(stream)
    if conns is None:
        print("Failed to load file")
        return
    for engine in conns:
        host = conns[engine]["host"]
        port = conns[engine]["port"]
        user = conns[engine]["user"]
        conn = f"{engine}://{host}:{port}"
        passwd = keyring.get_password(conn, user)
        if passwd is None:
            if list_mode:
                print(f"Password for {user}@{conn} is NOT set!")
            else:
                passwd = getpass.getpass(f"Enter password for {user}@{conn}: ")
                keyring.set_password(conn, user, passwd)
        else:
            print(f"Password for {user}@{conn} currently set to {passwd}")
            if list_mode:
                pass
            else:
                print("Change? [N/y]")
                ans = ""
                while ans.lower() not in ["y", "n", "\r"]:
                    ans = readchar.readkey()
                if ans.lower() == "y":
                    passwd = getpass.getpass(f"Enter password for {user}@{conn}: ")
                    keyring.set_password(conn, user, passwd)

if __name__ == "__main__":
    main()