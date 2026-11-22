import { act, cleanup, render } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { usePolling } from './usePolling';

function Probe({ fn, active }: { fn: () => Promise<void>; active: boolean }) {
  usePolling(fn, active);
  return null;
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => { cleanup(); vi.useRealTimers(); });

describe('usePolling', () => {
  it('calls every 3 seconds only while active', async () => {
    const fn = vi.fn().mockResolvedValue(undefined);
    const { rerender } = render(<Probe fn={fn} active />);
    await act(() => vi.advanceTimersByTimeAsync(6100));
    expect(fn).toHaveBeenCalledTimes(2);
    rerender(<Probe fn={fn} active={false} />);
    await act(() => vi.advanceTimersByTimeAsync(9000));
    expect(fn).toHaveBeenCalledTimes(2);
  });
  it('never overlaps calls', async () => {
    let release = () => {};
    const fn = vi.fn(() => new Promise<void>((r) => { release = r; }));
    render(<Probe fn={fn} active />);
    await act(() => vi.advanceTimersByTimeAsync(10000));
    expect(fn).toHaveBeenCalledTimes(1);
    release();
    await act(() => vi.advanceTimersByTimeAsync(3100));
    expect(fn).toHaveBeenCalledTimes(2);
  });
  it('stops on unmount', async () => {
    const fn = vi.fn().mockResolvedValue(undefined);
    const { unmount } = render(<Probe fn={fn} active />);
    unmount();
    await act(() => vi.advanceTimersByTimeAsync(9000));
    expect(fn).not.toHaveBeenCalled();
  });
});
