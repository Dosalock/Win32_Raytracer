
#ifndef UNICODE
#define UNICODE
#endif 

#include "main.h"

HBITMAP hBitmap = NULL;
HDC hdcWindow = NULL;
BYTE* lpvBits = NULL;
RECT window = {};
int height = 0;
int width = 0;

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);





int WINAPI WinMain(HINSTANCE hInstance, 
					HINSTANCE hPrevInstance, 
					LPSTR lpCmdLine, 
					int nCmdShow) { 

	//Register window class
	const wchar_t CLASS_NAME[] = L"PiskWindow";


	WNDCLASS WindowClass = {};
		
	WindowClass.lpfnWndProc = WindowProc;
	WindowClass.hInstance = hInstance;
	WindowClass.lpszClassName = CLASS_NAME;

	RegisterClass(&WindowClass);

	// Create window
	HWND hwnd = CreateWindowEx(
		0,
		CLASS_NAME,
		L"What Is This Text",
		WS_OVERLAPPEDWINDOW,		// Style of window

		CW_USEDEFAULT, CW_USEDEFAULT, 800, 800,

		NULL,
		NULL,
		hInstance,
		NULL
		);

	if (hwnd == NULL)
	{
		return 0;
	}

	GetClientRect(hwnd, &window);
	Init(&lpvBits, &window, &hBitmap);
	ShowWindow(hwnd, nCmdShow);
	UpdateWindow(hwnd);

	MSG msg = { };

	while (GetMessage(&msg, NULL, 0 , 0) > 0)
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
	
	return 0; 
}
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{

	
	width = window.right;
	height = window.bottom;
	switch (uMsg)
	{
		case WM_CREATE:
		{
			


			break;
		}
		case WM_DESTROY:
		{
			// Cleanup
			if (hBitmap) {
                DeleteObject(hBitmap);
            }
			PostQuitMessage(0);
		
			break;
		}

		case WM_PAINT:
		{
			PAINTSTRUCT ps;
			
			GetClientRect(hwnd, &window); // Use client area dimensions
			width = window.right;
			height = window.bottom;
			HDC hdc = BeginPaint(hwnd, &ps);
			
			HDC hdcMem = CreateCompatibleDC(hdc);
			HGDIOBJ oldBitmap = SelectObject(hdcMem, hBitmap);
			
			BitBlt(hdc, 0, 0, width, height, hdcMem, 0, 0, SRCCOPY);

			SelectObject(hdcMem, oldBitmap);
			DeleteDC(hdcMem);

			// All painting occurs here, between BeginPaint and EndPaint.

			EndPaint(hwnd, &ps);
		
			break;	
		}
		case WM_MOVE:
		{
			// TODO: do we want to do something on move?
		}
		case WM_KEYDOWN:
		{
			if (wParam == 'Q') {
				Draw(&lpvBits, width, height);
				InvalidateRect(hwnd, NULL, TRUE);
			}
			break;
		}
		default:
		{
			return DefWindowProc(hwnd, uMsg, wParam, lParam);
		}
	}
}




