import numpy as np
import matplotlib.pyplot as plt

def getAcc(pos, mass, G, softening):
    x = pos[:,0:1]
    y = pos[:,1:2]
    z = pos[:,2:3]

    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    inv_r3 = (dx**2 + dy**2 + softening**2)
    inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-1.5)

    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass

    a = np.hstack((ax,ay,az))

    return a

def getEnergy( pos, vel, mass, G ):
	"""
	Get kinetic energy (KE) and potential energy (PE) of simulation
	pos is N x 3 matrix of positions
	vel is N x 3 matrix of velocities
	mass is an N x 1 vector of masses
	G is Newton's Gravitational constant
	KE is the kinetic energy of the system
	PE is the potential energy of the system
	"""
	# Kinetic Energy:
	KE = 0.5 * np.sum(np.sum( mass * vel**2 ))


	# Potential Energy:

	# positions r = [x,y,z] for all particles
	x = pos[:,0:1]
	y = pos[:,1:2]
	z = pos[:,2:3]

	# matrix that stores all pairwise particle separations: r_j - r_i
	dx = x.T - x
	dy = y.T - y
	dz = z.T - z

	# matrix that stores 1/r for all particle pairwise particle separations 
	inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
	inv_r[inv_r>0] = 1.0/inv_r[inv_r>0]

	# sum over upper triangle, to count each interaction only once
	PE = G * np.sum(np.sum(np.triu(-(mass*mass.T)*inv_r,1)))
	
	return KE, PE

def getForce(mass, acc):
	'''Outputs an N x 1 vector of particle forces'''

	acc_mag = (acc[:,0]**2 + acc[:, 1]**2)**0.5 
	force = np.diag(mass * acc_mag)
	return force

# Creates a cluster of particles within a random range centered around a random (x, y) coordinate
def createCluster(N_div, universe_size, range_bounds):
    m_universe_size = universe_size * 10
    range = np.random.randint(low = range_bounds[0], high = range_bounds[1])
    boundary = m_universe_size - range
    center_x = np.random.randint(low = -boundary, high = boundary)
    center_y = np.random.randint(low = -boundary, high = boundary)
    pos_x = np.random.randint(size = (N_div,1), low = (center_x - range), high = (center_x + range))/10
    pos_y = np.random.randint(size = (N_div,1), low = (center_y - range), high = (center_y + range))/10
    t = np.zeros((N_div,1))
    return np.hstack((pos_x, pos_y, t))

def createCloud(N_div, universe_size):
    return np.random.randint(size = (N_div,3), low = -(universe_size*10), high = (universe_size*10))/10

def main():
	""" N-body simulation """
	
	# Simulation parameters
	N         = 1600                 # Number of particles
	divs      = 5                   # number of clusters
	N_div     = int(N/divs)         # number of particles per cluster
	t         = 0                   # current time of the simulation
	tEnd      = 2000.0              # time at which simulation ends
	dt        = 0.2                 # timestep
	softening = 1                 # softening length
	G         = 6.67*(10**-11)      # Newton's Gravitational Constant
	range_bounds = [10, 50]          # [min, max] of cluster size range
	p_size = 0.1
	universe_size = 50

	plotRealTime = True     # enable to plot in real time
	hasCloud = True         # enable to generate a diffuse particle cloud
	dynamic_size = False    # enable to visualize particle force magnitude by size
	dynamic_color = False   # enable to visualize particle force magnitude by color

	color_map = 'turbo'
	normalization = 'linear'
	
	# Generate initial conditions
	mass = (7.8*(10**8))*(np.random.randn(N,1)+1.2)  # randomly fill particle mass vector
	vel  = np.random.randn(N,3)/8  # randomly fill particle velocity vector

    # Generate initial particle positions
	pos = createCloud(N_div, universe_size) if hasCloud else createCluster(N_div, universe_size, range_bounds)  # generate diffused cloud of particles
	for i in range((divs-1)):
		cluster = createCluster(N_div, universe_size, range_bounds)
		pos = np.concatenate((pos, cluster), axis = 0)

	# Convert to Center-of-Mass frame
	vel -= np.mean(mass * vel,0) / np.mean(mass)
	
	# calculate initial gravitational accelerations
	acc = getAcc( pos, mass, G, softening )

    # calculate initial forces
	f = getForce(mass, acc)
	
	# calculate initial energy of system
	KE, PE = getEnergy( pos, vel, mass, G )
	
	# number of timesteps
	Nt = int(np.ceil(tEnd/dt))
	
	# save energies, particle orbits for plotting trails
	pos_save = np.zeros((N,3,Nt+1))
	pos_save[:,:,0] = pos
	KE_save = np.zeros(Nt+1)
	KE_save[0] = KE
	PE_save = np.zeros(Nt+1)
	PE_save[0] = PE
	t_all = np.arange(Nt+1)*dt

	colors = np.zeros((N, 1))
	
	# prep figure
	fig = plt.figure(figsize=(10,10), dpi=100)
	grid = plt.GridSpec(25, 1, wspace=0.0, hspace=0.3)
	ax1 = plt.subplot(grid[0:24,0])

	print(f)
	
	# Simulation Main Loop
	for i in range(Nt):
		# (1/2) kick
		vel += acc * dt/2.0
		
		# drift
		pos += vel * dt
		
		# update accelerations
		acc = getAcc( pos, mass, G, softening )
		
		# (1/2) kick
		vel += acc * dt/2.0
		
		# update time
		t += dt
		
		# get energy of system
		#KE, PE  = getEnergy( pos, vel, mass, G )

        # get forces on particles
		f = getForce(mass, acc)

		# detect border collision and "respawn"
		state = np.trunc(abs(pos[:,0:1] / universe_size))
		state += np.trunc(abs(pos[:,1:2] / universe_size))
		vel[:,0:2] -= state * vel[:,0:2]
		pos[:,0:1] -= state * (pos[:,0:1] + np.sin(t) * universe_size * 1.3)
		pos[:,1:2] -= state * (pos[:,1:2] + np.cos(t) * universe_size * 1.3)
		

		# update graph settings
		colors = abs(f) if dynamic_color else np.zeros((N,1))
		p_size = abs(f/(2*(10**9))) if dynamic_size else p_size
		
		# save energies, positions for plotting trail
		pos_save[:,:,i+1] = pos
		#KE_save[i+1] = KE
		#PE_save[i+1] = PE
		
		# plot in real time
		if plotRealTime or (i == Nt-1):
			plt.sca(ax1)
			plt.cla()
			#xx = pos_save[:,0,max(i-15,0):i+1]
			#yy = pos_save[:,1,max(i-15,0):i+1]
			#plt.scatter(xx,yy,s=0.001,color=[.7,.7,1])
			plt.scatter(pos[:,0], pos[:,1], s = p_size, c = colors, cmap = color_map, norm = normalization) # set color of particles
			ax1.set(xlim=(-universe_size, universe_size), ylim=(-universe_size, universe_size)) # set size of visible universe
			ax1.set_aspect('equal', 'box')
			ax1.set_xticks([-(universe_size),-(universe_size/2),0,(universe_size/2),(universe_size)])
			ax1.set_yticks([-(universe_size),-(universe_size/2),0,(universe_size/2),(universe_size)])
			
			plt.pause(0.001)

	# Save figure
	plt.savefig('nbody.png',dpi=240)
	plt.show()
	    
	return 0
	
if __name__== "__main__":
  main()