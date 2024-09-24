#include <stdio.h>
#include <conio.h>
#include <stdio.h>
#include <windows.h>


BOOL WINAPI Hook(DWORD event) {
    printf("Shutting down...");
    DWORD number = 0x00000000; 
    HKEY key;

    if (RegOpenKeyExW(HKEY_CURRENT_USER, L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Internet Settings", 0, KEY_SET_VALUE | KEY_WOW64_64KEY, &key) == ERROR_SUCCESS)
    {
        if (RegSetValueExW(key, L"ProxyEnable", 0, REG_DWORD, (LPBYTE)&number, sizeof(DWORD)) != ERROR_SUCCESS)
        {
            printf("Key not changed in registry \n");
            printf("Error %u ", (unsigned int)GetLastError());
        }
        RegCloseKey(key);
    }
    else 
    {
        printf("Unsuccessful in opening key  \n");
        printf("Cannot find key value in registry \n");
        printf("Error: %u ", (unsigned int)GetLastError());
    }
    printf("Finished.");
    exit(0);
    Sleep(300);
    return FALSE; // True means stop calling handlers, False calls the next handler.
}

int register_close_handler()
{
    SetConsoleCtrlHandler(Hook, TRUE);
}