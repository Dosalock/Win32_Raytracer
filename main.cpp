/**
 *
 *  @file      main.cpp
 *  @brief	   Application entry
 *  @author    Johan Karlsson - github.com/dosalock
 *  @date      8.11.2024
 *  @copyright Copyright © [2024] [Johan Karlsson]
 *
 */


/*------------------Includes---------------------*/
#include "main.h"
#include <unordered_map>
/*------------Varible initialzation--------------*/
#define clock std::chrono::high_resolution_clock

HBITMAP hBitmap                     = NULL;
HDC hdcWindow                       = NULL;
BYTE* lpvBits                       = NULL;
RECT window                         = { };
Camera cam                          = { };
int height                          = 0;
int width                           = 0;
std::chrono::time_point clock_start = std::chrono::steady_clock( ).now( );
bool cameraIsMoving                 = false;
bool drawing_frame                  = false;
float delta_time                    = 0.0f;
float moveSpeed                     = 0.1f;
float rotationSpeed                 = 2.0f;
long long secondsPerFrame           = 33'000'000;
Sphere scene[4]                     = { };
Light lights[3]                     = { };
std::unordered_map<int, bool> keyStates;

/*-------------Function Definitions--------------*/

void RenderFrame ( HWND hWnd )
{
	Draw( &lpvBits, width, height, cam, scene, lights);
	InvalidateRect( hWnd, NULL, TRUE );
}

void GameUpdate ( )
{
	// Camera updates should happen here
	if ( keyStates['W'] )
	{
		cam.MoveForward( moveSpeed );
	}
	if ( keyStates['A'] )
	{
		cam.MoveSideways( moveSpeed );
	}
	if ( keyStates['S'] )
	{
		cam.MoveForward( -moveSpeed );
	}
	if ( keyStates['D'] )
	{
		cam.MoveSideways( -moveSpeed );
	}
	if ( keyStates['Q'] )
	{
		cam.yaw -= rotationSpeed;
	}
	if ( keyStates['E'] )
	{
		cam.yaw += rotationSpeed;
	}
}

/**
 * @brief Application entrypoint.
 *
 * @param hInstance Handle to instance, base address of module memory
 * @param hPrevInstance Handle to previous instance - always NULL
 *		  if you need to detect if another exists, use CreateMutex
 *		  returns ERROR_ALREADY_EXISTS if theres already one named the same
 * @param lpCmdLine String to the command line for the application
 * @param nCmdShow Controls how the window is to be shown
 * @return
 */
int WINAPI WinMain ( _In_ HINSTANCE hInstance,
					 _In_opt_ HINSTANCE hPrevInstance,
					 _In_ LPSTR lpCmdLine,
					 _In_ int nCmdShow )
{
	// Register window class
	const wchar_t CLASS_NAME[] = L"Window";


	WNDCLASS WindowClass = { };

	WindowClass.lpfnWndProc   = WindowProc;
	WindowClass.hInstance     = hInstance;
	WindowClass.lpszClassName = CLASS_NAME;

	RegisterClass( &WindowClass );

	// Create window
	HWND hwnd = CreateWindowEx( 0,
								CLASS_NAME,
								L"Win32 Raytracer",
								WS_OVERLAPPEDWINDOW, // Style of window

								CW_USEDEFAULT,
								CW_USEDEFAULT,
								512,
								512,

								NULL,
								NULL,
								hInstance,
								NULL );

	if ( hwnd == NULL )
	{
		return 0;
	}
	CreateScene( scene, lights );
	GetClientRect( hwnd, &window );
	Init( &lpvBits, &window, &hBitmap );
	ShowWindow( hwnd, nCmdShow );
	UpdateWindow( hwnd );

	MSG msg = { };

	auto dEpoch      = std::chrono::steady_clock::now( );
	auto cached_time = std::chrono::steady_clock::now( );

	while ( msg.message != WM_QUIT )
	{
		if ( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE ) )
		{
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		else
		{
			// game loop
			auto now = std::chrono::steady_clock::now( );
			if ( ( now - clock_start ).count( ) >= secondsPerFrame )
			{
				clock_start = now;
				delta_time  = 1.0f; // Use this to scale camera speed
				GameUpdate( );
				drawing_frame = true;
			}
			if ( drawing_frame )
			{
				RenderFrame( hwnd );
				drawing_frame = false;
			}
		}
	}
	UnregisterClass( L"Win32 Raytracer", WindowClass.hInstance );
	return 0;
}

/**
 * Called from message dispatch, determines response to messages.
 *
 * @param hwnd handle to the window
 * @param uMsg message code; e.g. WM_KEYDOWN
 * @param wParam data pertaining to message e.g. which key pressen on
 * WM_KEYDOWN
 * @param lParam data pertaining to message if neeeded
 * @return
 */
LRESULT CALLBACK WindowProc ( HWND hwnd,
							  UINT uMsg,
							  WPARAM wParam,
							  LPARAM lParam )
{
	width  = window.right;
	height = window.bottom;

	switch ( uMsg )
	{
		case WM_CREATE:
		{
			break;
		}
		case WM_DESTROY:
		{
			// Cleanup
			if ( hBitmap )
			{
				DeleteObject( hBitmap );
			}
			PostQuitMessage( 0 );

			break;
		}

		case WM_PAINT:
		{
			PAINTSTRUCT ps;

			GetClientRect( hwnd, &window ); // Use client area dimensions

			width   = window.right;
			height  = window.bottom;
			HDC hdc = BeginPaint( hwnd, &ps );

			HDC hdcMem        = CreateCompatibleDC( hdc );
			HGDIOBJ oldBitmap = SelectObject( hdcMem, hBitmap );

			BitBlt( hdc, 0, 0, width, height, hdcMem, 0, 0, SRCCOPY );

			SelectObject( hdcMem, oldBitmap );
			DeleteDC( hdcMem );

			// All painting occurs here, between BeginPaint and EndPaint.

			EndPaint( hwnd, &ps );

			break;
		}
		case WM_MOVE:
		{
			// TODO: do we want to do something on move?
		}
		case WM_KEYUP:
		{
			keyStates[wParam] = false;
			break;
		}
		case WM_KEYDOWN:
		{
			keyStates[wParam] = true;
			break;
		}
		default:
		{
			return DefWindowProc( hwnd, uMsg, wParam, lParam );
		}
	}
	return DefWindowProc( hwnd, uMsg, wParam, lParam );
}
