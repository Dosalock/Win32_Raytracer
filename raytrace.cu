#include "raytrace.cuh"

/*------------Template Declarations---------------*/


__device__ Sphere* cuda_scene;
__device__ Light*  cuda_lights;

#define HEIGHT 1024
#define WIDTH  1024

void ExitCleanup ( )
{
    // cudaFree(cuda_scene);
    // cudaFree(lights);
}

__device__ bool IntersectionBounds ( float T, float t_min, float t_max )
{
    return ( T > t_min && T < t_max ); // Strict inequality
}

__device__ float4 ReflectRay ( const float4& R, const float4& N )
{
    return ( 2.0f * dot( N, R ) * N ) - R;
}

__device__ float
  CalcLight ( const float4& P, const float4& N, const float4& V, const int& s )
{
    float  intensity = 0.0;
    float  t_max     = 0;
    float4 L         = { };
    float4 R         = { };
    for ( int i = 0; i < sizeof( cuda_lights ) / sizeof( Light ); i++ ) {
        if ( cuda_lights[i].type == cuda_lights->ambient ) {
            intensity += cuda_lights[i].intensity;
        }
        else {
            if ( cuda_lights[i].type == cuda_lights->point ) {
                L     = ( cuda_lights[i].pos - P );
                t_max = 1;
            }
            else {
                L     = cuda_lights[i].pos;
                t_max = INFINITY;
            }
            L = L;
            auto [closest_sphere, closest_t] =
              ClosestIntersection( P, L, 0.00001, t_max );
            if ( closest_sphere != NULL ) {
                continue;
            }

            float n_dot_l = dot( N, L );
            if ( n_dot_l > 0 ) {
                intensity +=
                  cuda_lights[i].intensity * n_dot_l / ( len( N ) * len( L ) );
            }

            if ( s != -1 ) {
                R             = ReflectRay( L, N );
                float r_dot_v = dot( R, V );

                if ( r_dot_v > 0 ) {
                    intensity +=
                      cuda_lights[i].intensity
                      * pow( r_dot_v / ( len( R ) * ( len( V ) ) ), s );
                }
            }
        }
    }
    return intensity;
}

__device__ float IntersectRaySphere ( const float4& O,
                                      const float4& D,
                                      const Sphere& sphere,
                                      const double& dDot )
{
    float4 CO = { };
    CO        = O - sphere.center;

    float a = dDot;
    float b = 2 * dot( CO, D );
    float c = dot( CO, CO ) - sphere.sRadius;

    float discr = b * b - 4 * a * c;

    if ( discr < 0 ) {
        return INFINITY;
    }
    else if ( discr == 0 ) {
        return -b / ( 2 * a );
    }

    float t = ( -b - __fsqrt_rd( discr ) )
              / ( 2 * a ); // Minimize compute only go for 1 root;

    return t;
}

__device__ intersection ClosestIntersection ( const float4& O,
                                              const float4& D,
                                              const double& t_min,
                                              const double& t_max )
{
    float   closest_t      = INFINITY;
    Sphere* closest_sphere = NULL;
    float   d_dot_d        = dot( D, D ); // Cache immutable value


    for ( int i = 0; i < sizeof( cuda_scene ) / sizeof( Sphere ); i++ ) {
        double t = IntersectRaySphere( O, D, cuda_scene[i], d_dot_d );

        if ( IntersectionBounds( t, t_min, t_max ) && t < closest_t ) {
            closest_t      = t;
            closest_sphere = const_cast<Sphere*>( &cuda_scene[i] );
        }
    }
    return intersection( closest_sphere, closest_t );
}

__device__ COLORREF TraceRay ( const float4& O,
                               const float4& D,
                               const float&  t_min,
                               const float&  t_max,
                               const int&    recursionDepth )
{
    float4 N = { };
    float4 P = { };
    float4 R = { };

    auto [closest_sphere, closest_t] = ClosestIntersection( O, D, t_min, t_max );

    if ( closest_sphere == NULL ) {
        return RGB( 0, 0, 0 );
    }

    P = O + ( closest_t * D );
    N = normalize( P - closest_sphere->center );

    float res = CalcLight( P, N, invert( D ), closest_sphere->specularity );
    int   r   = ( int )round( GetRValue( closest_sphere->color ) * res );
    int   g   = ( int )round( GetGValue( closest_sphere->color ) * res );
    int   b   = ( int )round( GetBValue( closest_sphere->color ) * res );

    float refl = closest_sphere->reflective;

    if ( recursionDepth <= 0 || refl <= 0 ) {
        return RGB( max( 0, min( 255, r ) ),
                    max( 0, min( 255, g ) ),
                    max( 0, min( 255, b ) ) );
    }


    R = ReflectRay( invert( D ), N );
    COLORREF reflectedColor =
      TraceRay( P, R, t_min, t_max, recursionDepth - 1 );

    int reflected_r = ( int )roundf( GetRValue( reflectedColor ) ) * refl;
    int reflected_g = ( int )roundf( GetGValue( reflectedColor ) ) * refl;
    int reflected_b = ( int )roundf( GetBValue( reflectedColor ) ) * refl;


    return RGB(
      max( 0, min( 255, static_cast<int>( r * ( 1 - refl ) + reflected_r ) ) ),
      max( 0, min( 255, static_cast<int>( g * ( 1 - refl ) + reflected_g ) ) ),
      max( 0,
           min( 255, static_cast<int>( b * ( 1 - refl ) + reflected_b ) ) ) );
}

__device__ float4 CanvasToViewPort ( const int& x, const int& y )
{
    // for simplicity : Vw = Vh = d = 1    approx 53 fov
    float aspectRatio = static_cast<float>( WIDTH ) / HEIGHT;

    // x and y to the viewport, adjusting by aspect ratio
    float fovMod = 1;
    float viewportX =
      ( x - WIDTH / 2.0 ) * ( ( 1.0 * fovMod ) / WIDTH ) * aspectRatio;
    float viewportY =
      -( y - HEIGHT / 2.0 )
      * ( ( 1.0 * fovMod ) / HEIGHT ); // Flip Y to match 3D space orientation

    return make_float4( viewportX,
                        viewportY,
                        1,
                        0 ); // Z=1 for perspective projection
}

__global__ void cuda_Draw ( BYTE* pLpvBits, Camera& cam )
{
    float4 D              = { };
    float4 N              = { };
    float4 P              = { };
    float  t_min          = 0.0001;
    float  t_max          = INFINITY;
    int    recursionDepth = 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    D = normalize( CanvasToViewPort( x, y ) );
    D = normalize(
      apply_camera_rotation( cam.position, cam.yaw, cam.pitch, cam.roll ) );
    COLORREF color = TraceRay( cam.position, D, t_min, t_max, recursionDepth );
    D              = normalize( D );


    int offset = ( y * WIDTH + x ) * 4;
    if ( offset >= 0 && offset < WIDTH * HEIGHT * 4 - 4 ) {
        pLpvBits[offset + 0] = ( int )GetBValue( color );
        pLpvBits[offset + 1] = ( int )GetGValue( color );
        pLpvBits[offset + 2] = ( int )GetRValue( color );
        pLpvBits[offset + 3] = 255;
    }
}

__host__ void
  Draw_Caller ( BYTE** pLpvBits, Camera& cam, Sphere *scene, Light *lights )
{
    int buffer_size = WIDTH * HEIGHT * sizeof( BYTE ) * 4;

    int N = 1024;

    dim3 threadsPB( 16, 16 );
    dim3 numB( N / threadsPB.x, N / threadsPB.y );

    BYTE*  cuda_lpvbits;
    size_t src_pitch =
      ( ( WIDTH * 4 + 3 )
        & ~3 ); // AND with (NOT 3) ensures last two digits are always 0
    size_t dest_pitch;

    cudaMallocPitch(
      &cuda_lpvbits,
      &dest_pitch,
      WIDTH * 4 * sizeof( BYTE ), // 4 bytes for each pixel; R, G, B, alpha
      HEIGHT ); // number of rows

    cudaMemcpy2D( cuda_lpvbits, // Destinaion
                  dest_pitch,
                  *pLpvBits, // Source
                  src_pitch,
                  WIDTH * 4 * sizeof( BYTE ),
                  HEIGHT,
                  cudaMemcpyHostToDevice );


    Camera* cuda_cam;

    cudaMalloc( &cuda_cam, sizeof( Camera ) );
    cudaMemcpy( cuda_cam, &cam, sizeof( Camera ), cudaMemcpyHostToDevice );


    int number_of_spheres = 3;
    int number_of_lights  = 2;

    cudaMalloc( &cuda_scene, ( number_of_spheres * sizeof( Sphere ) ) );
    cudaMalloc( &cuda_lights, ( number_of_lights * sizeof( Light ) ) );

    cudaMemcpyToSymbol( cuda_scene,
                        scene,
                        number_of_spheres * sizeof( Sphere ),
                        0,
                        cudaMemcpyHostToDevice );

    cudaMemcpyToSymbol( cuda_lights,
                        lights,
                        number_of_lights * sizeof( Light ),
                        0,
                        cudaMemcpyHostToDevice );


    cuda_Draw<<<numB, threadsPB>>>( cuda_lpvbits, *cuda_cam );

    cudaDeviceSynchronize( );

    cudaMemcpy2D( *pLpvBits, // Destination
                  src_pitch,
                  cuda_lpvbits, // Source
                  dest_pitch,
                  WIDTH * 4 * sizeof( BYTE ),
                  HEIGHT,
                  cudaMemcpyDeviceToHost );


    cudaFree( cuda_cam );
    cudaFree( cuda_lpvbits );
}
