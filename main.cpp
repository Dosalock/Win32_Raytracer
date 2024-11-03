
#ifndef UNICODE
#define UNICODE
#endif 

#include "Windows.h"


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

		CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,

		NULL,
		NULL,
		hInstance,
		NULL
		);

	if (hwnd == NULL)
	{
		return 0;
	}

	ShowWindow(hwnd, nCmdShow);

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
	switch (uMsg)
	{
	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;

	case WM_PAINT:
	{
		PAINTSTRUCT ps;
		HBRUSH hbr = CreateSolidBrush(RGB(0, 255, 0, 0));
		HDC hdc = BeginPaint(hwnd, &ps);

		// All painting occurs here, between BeginPaint and EndPaint.

		FillRect(hdc, &ps.rcPaint, hbr);

		EndPaint(hwnd, &ps);
	}
	return 0;

	}
	return DefWindowProc(hwnd, uMsg, wParam, lParam);
}
	

