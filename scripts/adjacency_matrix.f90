subroutine adjacency_matrix(labels, adjmat, num_components)
    integer, intent(in) :: num_components
    integer(8), dimension(:, :), intent(in) :: labels
    integer(8), dimension(num_components, num_components), intent(inout) :: adjmat

    integer(8) :: src, dst
    nx = size(labels,1)
    ny = size(labels,2)

    ! Let's be careful, labels start at zero, when invoking labels, always add 1

    adjmat = 0
    ! We travel the whole image to find neighbours
    do i = 2, nx-1
        do j = 2, ny-1
            src = labels(i, j) + 1
            dst = labels(i-1, j) + 1
            adjmat(src, dst) = adjmat(src, dst) + 1
            adjmat(dst, src) = adjmat(dst, src) + 1
            dst = labels(i+1, j) + 1
            adjmat(src, dst) = adjmat(src, dst) + 1
            adjmat(dst, src) = adjmat(dst, src) + 1
            dst = labels(i, j+1) + 1
            adjmat(src, dst) = adjmat(src, dst) + 1
            adjmat(dst, src) = adjmat(dst, src) + 1
            dst = labels(i, j-1) + 1
            adjmat(src, dst) = adjmat(src, dst) + 1
            adjmat(dst, src) = adjmat(dst, src) + 1
        end do
    end do
    print *, "[Fortran] Computed adjacency matrix"
end subroutine

subroutine color_distance_matrix(picture, labels, adjmat, distmat, mean_colors)
    use omp_lib
    implicit none
    real, dimension(:,:,:), intent(in) :: picture
    integer(8), dimension(:,:), intent(in) :: labels
    integer(8), dimension(:,:), intent(in) :: adjmat
    real, dimension(:,:), intent(inout) :: distmat

    real, dimension(size(adjmat, 1), 4), intent(inout) :: mean_colors
    logical(1), dimension(size(picture,1), size(picture, 2)) :: mask
    integer(8) :: N, i, j, num_threads
    real :: area

    N = size(adjmat, 1)
    print *, "Compute segments' mean color"
    ! Compute mean color of each segment
    !$OMP PARALLEL DO private(mask,area,i) shared(mean_colors,labels,picture) firstprivate(N)
    do i = 1_8, N
        if (all(adjmat(i,:) .eq. 0)) then
            cycle
        endif
        if (i == 1) then
            ! The if clause can be removed for serious use.
            ! It is here for debugging only.
            num_threads = OMP_get_num_threads()
            print *, 'num_threads running:', num_threads
        end if
        mask = labels .eq. i - 1
        area = real(count(mask))
        if (area > 0.0_4) then
            mean_colors(i, 1) = sum(picture(:, :, 1), mask=mask) / area 
            mean_colors(i, 2) = sum(picture(:, :, 2), mask=mask) / area
            mean_colors(i, 3) = sum(picture(:, :, 3), mask=mask) / area
            mean_colors(i, 4) = area
        else
            mean_colors(i, :) = 0.0_4
        end if
    end do
    !$OMP END PARALLEL DO

    distmat = 0.0_4
    print *, "Building distance matrix"
    do i = 1_8, N
        do j = 1_8, N
            if (adjmat(i,j) > 0) then
                distmat(i,j) = sqrt(sum(abs(mean_colors(i, 1:3) - mean_colors(j, 1:3))**2))
            endif
        end do
    end do
    print *, "Fortran finished."
end subroutine

subroutine rag_merge(adjmat, distmat, mapping, distance_threshold, mean_colors)
    integer(8), dimension(:,:), intent(inout) :: adjmat
    real, dimension(:,:), intent(inout) :: distmat
    integer(8), dimension(:), intent(inout) :: mapping
    real, dimension(size(adjmat, 1), 4), intent(inout) :: mean_colors

    integer(8) :: N, k, i, j
    real, intent(in) :: distance_threshold

    N = size(distmat, 1)

    do i = 1_8, N
        do j = 1_8, i
            if (adjmat(i,j) > 0 .and. i .ne. j) then
                if(distmat(i,j) .lt. distance_threshold) then
                    mapping(i) = j - 1
                    
                    
                    do k = 1, size(mapping,1)
                        if (mapping(k) .eq. i - 1) then
                            mapping(k) = j -1
                        end if
                    end do
                    adjmat(j, :) = adjmat(j, :) + adjmat(i, :)
                    adjmat(:, j) = adjmat(:, j) + adjmat(:, i)
                    adjmat(i, :) = 0
                    adjmat(:, i) = 0

                    mean_colors(j,1:3) = (mean_colors(i, 4) * mean_colors(i, 1:3) + mean_colors(j, 1:3) * mean_colors(j, 4)) 
                    mean_colors(j, 1:3) = mean_colors(j, 1:3) / (mean_colors(i, 4)+mean_colors(j, 4))
                    do k = 1_8, N
                        if (adjmat(j, k) .gt. 0) then
                            distmat(j, k) = sqrt(sum(abs(mean_colors(j,1:3)-mean_colors(k,1:3))**2))
                            distmat(k, j) = distmat(j, k)
                        end if
                    end do
                end if
            end if
        end do
    end do
end subroutine